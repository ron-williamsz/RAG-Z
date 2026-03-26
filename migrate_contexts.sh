#!/bin/bash
# ============================================================
# migrate_contexts.sh
# Migra contextos do servidor origem para o destino
# Executar da máquina que alcança AMBOS os servidores
# ============================================================

ORIGEM="http://192.168.0.74:8002"
DESTINO="http://72.60.151.187:8002"
TMP_DIR="/tmp/rag_migration"
LOG_FILE="$TMP_DIR/migration.log"

# ---- cores ----
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p "$TMP_DIR"
echo "" > "$LOG_FILE"

log() { echo -e "$1" | tee -a "$LOG_FILE"; }

# ---- verificar conectividade ----
log "\n=== Verificando conectividade ==="
if ! curl -sf --connect-timeout 5 "$ORIGEM/health" > /dev/null; then
    log "${RED}ERRO: Origem $ORIGEM não acessível${NC}"
    exit 1
fi
log "${GREEN}Origem OK${NC}: $ORIGEM"

if ! curl -sf --connect-timeout 5 "$DESTINO/health" > /dev/null; then
    log "${RED}ERRO: Destino $DESTINO não acessível${NC}"
    exit 1
fi
log "${GREEN}Destino OK${NC}: $DESTINO"

# ---- verificar se endpoints de export/import existem ----
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$ORIGEM/api/contexts/default/export")
if [ "$STATUS" = "404" ] && curl -s "$ORIGEM/api/contexts/default/export" | grep -q "não possui índice"; then
    : # 404 esperado para contexto sem índice - endpoint existe
elif [ "$STATUS" = "404" ]; then
    log "${RED}ERRO: Endpoint de export não existe na origem. Faça o pull e rebuild lá primeiro.${NC}"
    exit 1
fi

# ---- listar contextos com dados na origem ----
log "\n=== Listando contextos com dados na origem ==="
CONTEXTS_JSON=$(curl -s "$ORIGEM/api/contexts")
if [ -z "$CONTEXTS_JSON" ]; then
    log "${RED}ERRO: Falha ao listar contextos da origem${NC}"
    exit 1
fi

# filtra apenas contextos com documentos
WITH_DATA=$(echo "$CONTEXTS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ctxs = [c['name'] for c in data.get('contexts', []) if c.get('total_documents', 0) > 0]
print('\n'.join(ctxs))
")

TOTAL=$(echo "$WITH_DATA" | grep -c . || true)
log "Contextos com dados: ${YELLOW}$TOTAL${NC}"

if [ "$TOTAL" -eq 0 ]; then
    log "${YELLOW}Nenhum contexto com dados encontrado na origem.${NC}"
    exit 0
fi

# ---- migrar ----
log "\n=== Iniciando migração ==="
SUCCESS=0
SKIP=0
FAIL=0

while IFS= read -r CTX; do
    [ -z "$CTX" ] && continue

    EXPORT_FILE="$TMP_DIR/${CTX}.json"

    printf "%-30s" "  $CTX"

    # exportar da origem
    HTTP_STATUS=$(curl -s -o "$EXPORT_FILE" -w "%{http_code}" "$ORIGEM/api/contexts/$CTX/export")

    if [ "$HTTP_STATUS" != "200" ]; then
        log " ${YELLOW}SKIP${NC} (origem retornou $HTTP_STATUS)"
        ((SKIP++))
        continue
    fi

    DOC_COUNT=$(python3 -c "import json; d=json.load(open('$EXPORT_FILE')); print(d.get('total_documents',0))" 2>/dev/null || echo 0)
    if [ "$DOC_COUNT" -eq 0 ]; then
        log " ${YELLOW}SKIP${NC} (0 documentos)"
        ((SKIP++))
        continue
    fi

    # importar no destino
    IMPORT_RESULT=$(curl -s -X POST "$DESTINO/api/contexts/$CTX/import" \
        -H "Content-Type: application/json" \
        -d @"$EXPORT_FILE")

    if echo "$IMPORT_RESULT" | grep -q '"status":"success"\|"indexed"'; then
        INDEXED=$(echo "$IMPORT_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('indexed_documents', d.get('total_documents','?')))" 2>/dev/null || echo "?")
        log " ${GREEN}OK${NC} ($DOC_COUNT docs → $INDEXED indexados)"
        ((SUCCESS++))
    else
        ERR=$(echo "$IMPORT_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('detail','erro desconhecido')[:80])" 2>/dev/null || echo "erro")
        log " ${RED}FAIL${NC} ($ERR)"
        ((FAIL++))
    fi

    # limpar arquivo temporário
    rm -f "$EXPORT_FILE"

done <<< "$WITH_DATA"

# ---- resumo ----
log "\n=== Resumo ==="
log "  ${GREEN}Sucesso:${NC}  $SUCCESS"
log "  ${YELLOW}Ignorados:${NC} $SKIP"
log "  ${RED}Falhas:${NC}    $FAIL"
log "  Log salvo em: $LOG_FILE"

# ---- verificar destino ----
log "\n=== Verificando destino após migração ==="
curl -s "$DESTINO/api/contexts" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ctxs = data.get('contexts', [])
with_data = [c for c in ctxs if c.get('total_documents', 0) > 0]
print(f'Total contextos no destino: {len(ctxs)}')
print(f'Com documentos:             {len(with_data)}')
"
