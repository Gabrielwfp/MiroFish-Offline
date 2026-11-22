# Informe técnico — Integración de oMLX en MiroFish-Offline

Fecha: 2026-11-21 · Entorno: macOS (Apple Silicon), oMLX 0.8.2, Docker Desktop 29.9.3, Neo4j 5.18.1 (solo en laboratorio)
Alcance: solo local. No se hizo commit ni push, no se tocó `.env` y no se borraron datos ni volúmenes.

**Dictamen: NO-GO para validación de extremo a extremo con la configuración actual. GO CON CONDICIONES para la capa de integración (código, configuración, Docker, Neo4j).** Detalle en §10.

---

## 1. Archivos modificados y justificación

| Archivo | Cambio | Motivo |
|---|---|---|
| `backend/app/storage/embedding_service.py` | Cambia el endpoint `/api/embed` de Ollama por `{base}/embeddings` en formato OpenAI. Envía `Authorization: Bearer` solo si hay clave. Valida la dimensión de cada vector. Errores 4xx sin reintento y con el cuerpo del error. Respuestas malformadas (incluidos `TypeError` y `AttributeError`) se convierten en `EmbeddingError`. | oMLX solo expone `/v1/embeddings` y exige autenticación. Una dimensión incorrecta se guardaría en Neo4j sin indexarse (verificado, §5). |
| `backend/app/config.py` | Nuevas variables `EMBEDDING_API_KEY` (si falta, usa `LLM_API_KEY`) y `EMBEDDING_DIMENSIONS` (por defecto 768). `EMBEDDING_BASE_URL` pasa a incluir `/v1`. | Configurar el proveedor de embeddings por separado del LLM, manteniendo la compatibilidad con 768. |
| `backend/app/storage/neo4j_schema.py` | Los índices `entity_embedding` y `fact_embedding` toman la dimensión de `Config.EMBEDDING_DIMENSIONS`. | El 768 estaba fijo en el código. |
| `.env.example` | Plantilla para oMLX (puerto 11436, `/v1`, dimensión), variantes para Docker con `host.docker.internal`, nota sobre la migración de índices y comentario correcto sobre `OPENAI_*`. | Documentar una configuración reproducible. |
| `docker-compose.yml` | Se quita `depends_on: ollama` (el servicio ya no existía) y se añade `extra_hosts: host.docker.internal:host-gateway`. | `docker compose up` fallaba. Con la entrada nueva, el contenedor alcanza el servidor del host también en Linux. |
| `backend/tests/test_embedding_service.py` (nuevo) | 18 pruebas unitarias con mocks. | No existía ninguna suite. |
| `CLAUDE.md` (nuevo) | Guía del repositorio con las incompatibilidades verificadas. | — |

## 2. Problemas encontrados y causas

1. **Embeddings: el modelo Jina no es servible por oMLX.** `POST /v1/embeddings` devuelve 400: «Model 'jina-embeddings-v5-text-small-retrieval-mlx' is not an embedding model». La causa está en `omlx/model_discovery.py`: `qwen3` es un `model_type` ambiguo y solo se clasifica como embedding si `architectures` contiene `Qwen3ForCausalLM`. El `config.json` del modelo no tiene `architectures`, así que oMLX lo trata como `llm`. Además, el modelo trae su propio `model.py`, con pooling del último token y prefijos `Query:`/`Document:`, que oMLX no usa.
2. **Tu `.env` no apunta al endpoint correcto.** `EMBEDDING_BASE_URL=http://10.0.0.13:11436` no lleva `/v1`, lo que da 404 (verificado). Además falta `EMBEDDING_DIMENSIONS`.
3. **NER incompatible con el LLM vía oMLX (nuevo, bloqueante).** `Qwen3.8-27B-oQ4e-mtp` responde al prompt de NER con una lista JSON en lugar de `{"entities":[...],"relations":[...]}`, aunque el cliente envía `response_format: json_object`, que oMLX no está forzando. `ner_extractor.py:155` falla con `'list' object has no attribute 'get'`, y tras 3 intentos queda 0 entidades.
4. **Fallos silenciosos previos.** `Neo4jStorage.add_text` (`neo4j_storage.py:202-206`) captura los errores de embeddings y guarda las entidades sin vector. El NER fallido también deja el chunk como «procesado». El build del grafo termina sin error visible, pero vacío.
5. **`uv.lock` desincronizado.** El lock es anterior al `pyproject.toml` del fork: la raíz sigue siendo `mirofish-backend` 0.1.0, `neo4j` solo aparece como dependencia transitiva de `camel-oasis` y queda `zep-cloud`. Por eso `uv run`/`uv sync` lo reescriben y `--frozen` falla.
6. **Fallo de `tiktoken`.** `pyproject.toml` declara `>=3.11` sin límite superior, y `uv` eligió el Python 3.15 de Homebrew. `tiktoken 0.7.0` (fijado por `camel-ai`) solo publica wheels hasta cp312, así que intenta compilar desde el código fuente con Rust, que no está instalado.
7. **`docker-compose.yml` roto en el árbol de trabajo:** `depends_on: ollama` apuntaba a un servicio eliminado.

## 3. Correcciones realizadas

- Puntos 2 (parte de código), 7 y la validación de dimensión: corregidos (§1).
- Bug encontrado por las pruebas: con `{"data": ["texto"]}` se escapaba un `AttributeError` sin capturar. Corregido.
- **No corregido, por decisión o por alcance:** punto 1 (requiere elegir otro modelo o modificar archivos del modelo), punto 3 (requiere cambiar el parser o el prompt de NER, o el modelo), punto 4 (cambia la semántica del build), punto 5 (reescribe `uv.lock`, requiere autorización; análisis de impacto en §4) y tu `.env`.

## 4. Versiones y dependencias verificadas

- **oMLX 0.8.2**, servidor gestionado en `0.0.0.0:11436`, autenticación obligatoria (`skip_api_key_verification=false`, `allow_unauthenticated_inference=false`).
- **Modelos (`/v1/models`):** Qwen3.8-27B-oQ4e-mtp, gemma-4-31B/26B, Muse-Glimmer-30B y Ornith-1.5-35B (todos `vlm`), y jina-embeddings-v5-text-small-retrieval-mlx (`llm`).
- **Dimensión real de Jina v5 small:** 1024, medida ejecutando el `model.py` del propio modelo con el MLX de oMLX. Vectores normalizados (norma ≈ 1,0) y semántica coherente: cos(consulta, documento relevante) = 0,554 frente a 0,105 con uno irrelevante. Es compatible con la similitud coseno, pero no se puede servir por oMLX (problema 1).
- **Python:** 3.11 es un **requisito real**, no un parche: `camel-oasis 0.2.5` declara `requires_python <3.12,>=3.10` y `camel-ai 0.2.78` declara `<3.13` (metadatos de PyPI).
- **Entorno aislado (venv 3.11 en el scratchpad, sin tocar el repo):** flask 3.1.2, neo4j (driver) 5.23.0, camel-ai 0.2.78, camel-oasis 0.2.5, tiktoken 0.7.0 (instalado desde wheel), openai 1.109.1, requests 2.32.5, pytest 8.2.0.
- **Análisis de impacto de regenerar el lock** con `requires-python = ">=3.11,<3.12"`, hecho en una copia: **0 cambios de versión**. Desaparecen `zep-cloud`, `setuptools` y el nombre viejo de la raíz, y `neo4j` pasa a ser dependencia directa.

## 5. Pruebas ejecutadas

**Unitarias, con mocks y sin red: 18/18 PASS**
`cd backend && <venv 3.11>/bin/python -B -m pytest tests -q -p no:cacheprovider`
Cubren:
- Formato OpenAI y URL, y Bearer presente o ausente.
- Errores 400, 401, 403 y 404 sin reintento.
- Errores 503 con reintento.
- Agotamiento de reintentos por conexión.
- Orden por `index`, caché y textos vacíos.
- Dimensión y cantidad incorrectas.
- 4 respuestas malformadas.
- Texto vacío.
- Configuración (`EMBEDDING_API_KEY` y su fallback, `EMBEDDING_DIMENSIONS` con su valor por defecto 768).
- Schema con 1024.

**Integración real con oMLX** (credenciales inyectadas desde `.env` por `config.py`, nunca impresas)

| Prueba | Resultado |
|---|---|
| `GET /v1/models` sin clave | 401: autenticación activa (PASS) |
| `GET /v1/models` con clave | 200 (PASS) |
| `EmbeddingService` con la base de tu `.env` (sin `/v1`) | `EmbeddingError` 404, el cliente lo reporta bien (PASS del cliente) |
| `EmbeddingService` con `/v1` y el modelo Jina | `EmbeddingError` 400 «not an embedding model» (**FAIL de compatibilidad**) |
| `EmbeddingService` con un modelo inexistente | `EmbeddingError` 404 (PASS del cliente) |
| `LLMClient.chat` con Qwen3.8-27B | Respuesta correcta en 1,9 s (PASS) |
| `LLMClient.chat_json` con un JSON simple | `{'ok': True, 'n': 3}` (PASS) |
| `NERExtractor.extract` | El modelo devuelve una lista y se extraen 0 entidades (**FAIL**) |

**Docker → oMLX** (contenedor `python:3.11-slim`, sin credenciales; un 401 prueba que se llega al servidor)

| Destino | Resultado |
|---|---|
| `host.docker.internal:11436` | 401 (alcanzable, PASS) |
| `10.0.0.13:11436` | 401 (alcanzable) |
| `localhost:11436` | Connection refused (esperado: es el propio contenedor) |

`docker compose config`: válido.

**Neo4j 5.18.1 en laboratorio aislado: 9/9 PASS**
Contenedor desechable, solo en 127.0.0.1:17687, sin volumen y ya eliminado. Script: `scratchpad/neo4j_lab_test.py`, que usa el schema y el Cypher reales de la app.
- El schema crea `entity_embedding` y `fact_embedding` con 1024.
- La búsqueda vectorial de nodos y de relaciones funciona (consultas reales de `search_service`).
- Una consulta con un vector de 768 sobre el índice de 1024 es rechazada (`ProcedureCallFailed`).
- Un nodo con vector de 768 **se guarda pero no se indexa**: fallo silencioso, y por eso la validación en el cliente.
- Volver a ejecutar el schema con otra dimensión **no cambia** los índices (`IF NOT EXISTS`).
- `DROP INDEX` y la recreación aplican la dimensión nueva; los nodos quedan intactos.
- Rollback: al recrear los índices de 1024 se reindexan los vectores existentes.

**Regresión (app contra el Neo4j de laboratorio)**
- `compileall` de `app/` y `scripts/`: OK.
- `create_app()` arranca con `Neo4jStorage`.
- `GET /health`, `/api/graph/project/list`, `/api/simulation/list` y `/api/report/list` devuelven 200.
- Importación de servicios: OK.
- `add_text` real termina «bien» con 0 entidades (problemas 3 y 4).

## 6. Pruebas bloqueadas

- **Embeddings reales de extremo a extremo con oMLX:** BLOQUEADA POR INCOMPATIBILIDAD DEL MODELO, no por credenciales. Hace falta un modelo de embeddings que oMLX reconozca (§9).
- **Build completo del grafo, simulación OASIS y report:** BLOQUEADO por el NER (problema 3). No se ejecutaron.
- **Conectividad desde el contenedor de la app** (`docker compose up` completo): no ejecutada; solo se probó con un contenedor equivalente. El contenedor de la app usa `npm run dev` e instala dependencias pesadas.
- **Linux con `host-gateway`:** no hay host Linux disponible. La configuración se validó con `docker compose config`.

## 7. Riesgos pendientes

1. **NER con Qwen3.8 vía oMLX (alto):** sin entidades no hay grafo, perfiles ni simulación.
2. **Fallos silenciosos en `add_text` (alto):** cualquier fallo de embeddings o NER produce grafos vacíos o sin vectores sin que el usuario lo vea.
3. **Licencia:** `jina-embeddings-v5-text-small-retrieval` es **CC-BY-NC-4.0** (no comercial). Revísala si el uso es comercial.
4. **Exposición de red:** oMLX escucha en `0.0.0.0` (`server_aliases` incluye dos IPs de la LAN) y queda accesible desde la red local, protegido solo por la API key. Si la app solo corre en este Mac, conviene enlazarlo a 127.0.0.1: Docker Desktop seguiría alcanzándolo por `host.docker.internal`. No lo cambié (es configuración global de oMLX).
5. **URLs con IP de la LAN en `.env` (`10.0.0.13`):** se rompen si cambia la IP asignada por DHCP. Usa `localhost` desde el host y `host.docker.internal` desde Docker.
6. **`uv.lock` desincronizado:** cualquier `uv run`/`uv sync` lo modifica, y la imagen Docker (`uv sync` en el Dockerfile) resuelve sin el lock real.
7. **Prefijos de consulta y documento:** los modelos de recuperación (Jina v5, Qwen3-Embedding) rinden mejor con instrucción o prefijo para la consulta. La app no los añade; la calidad de búsqueda no se validó.

## 8. Migración de Neo4j y rollback

**Estado actual en esta máquina:** no existe ningún volumen de Neo4j ni contenedor de MiroFish (`docker volume ls` sin coincidencias, nada en el 7687). **No hay embeddings previos que migrar ni que mezclar.** El procedimiento aplica cuando ya exista una base con datos. Todos los pasos destructivos requieren tu autorización explícita.

**Precondiciones**
1. Fijar el modelo de embeddings definitivo y medir su dimensión real con una petición a `/v1/embeddings`.
2. Inventario de dimensiones y modelos, en solo lectura:
   ```cypher
   SHOW INDEXES YIELD name, type, options WHERE type = 'VECTOR' RETURN name, options.indexConfig;
   MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN size(n.embedding) AS dim, count(*) AS n;
   MATCH ()-[r:RELATION]->() WHERE r.fact_embedding IS NOT NULL RETURN size(r.fact_embedding) AS dim, count(*) AS n;
   ```
   Si aparece más de una dimensión, o la dimensión coincide pero el modelo cambió (por ejemplo, 1024 de Jina frente a 1024 de Qwen3-Embedding), **los espacios vectoriales son incompatibles**: hay que regenerar *todos* los embeddings, no solo los de dimensión distinta.
3. Respaldo (Neo4j Community requiere la base detenida):
   ```bash
   docker compose stop neo4j
   docker run --rm -v mirofish-offline_neo4j_data:/data -v "$PWD/neo4j-backup":/backups \
     neo4j:5.18-community neo4j-admin database dump neo4j --to-path=/backups
   docker compose start neo4j
   ```
   También se puede copiar `backend/uploads/` (proyectos y documentos fuente).

**Migración** (destructiva; requiere autorización)
1. En `.env`: `EMBEDDING_MODEL=<nuevo>`, `EMBEDDING_DIMENSIONS=<dim real>`, `EMBEDDING_BASE_URL=...:11436/v1`.
2. `DROP INDEX entity_embedding IF EXISTS; DROP INDEX fact_embedding IF EXISTS;` (no borra datos; verificado en laboratorio).
3. Reiniciar el backend: `Neo4jStorage.__init__` ejecuta `_ensure_schema()` y recrea los índices con la dimensión nueva.
4. Regenerar los embeddings. El código no tiene una herramienta de re-embedding, así que hay dos opciones: (a) reconstruir cada grafo desde su documento fuente (borra y recrea el grafo, que es destructivo), o (b) escribir un script que recorra `Entity.summary` y `RELATION.fact` y reescriba los vectores con `EmbeddingService` (no destructivo para la estructura del grafo; no existe todavía).
5. Verificar con las consultas de inventario: una sola dimensión, con el total esperado.

**Rollback**
- Si solo se cambiaron los índices: restaurar `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` anteriores, `DROP INDEX` los dos índices y reiniciar el backend. Los índices se recrean y reindexan los vectores existentes (verificado en laboratorio).
- Si se regeneraron los embeddings o los grafos: `docker compose stop neo4j`, luego `neo4j-admin database load neo4j --from-path=/backups --overwrite-destination=true` (mismo patrón de `docker run` con el volumen), y por último restaurar el `.env` anterior y arrancar.

## 9. Configuración recomendada para oMLX

Valores para `.env`, sin credenciales; tu clave va solo en tu `.env`:
```env
LLM_API_KEY=<clave oMLX>
LLM_BASE_URL=http://localhost:11436/v1          # Docker: http://host.docker.internal:11436/v1
LLM_MODEL_NAME=<modelo que pase la prueba de NER>
EMBEDDING_BASE_URL=http://localhost:11436/v1    # imprescindible el /v1
EMBEDDING_MODEL=<modelo de embeddings detectado como "embedding" en /v1/models>
EMBEDDING_DIMENSIONS=<dimensión medida>
# EMBEDDING_API_KEY=   # opcional; por defecto usa LLM_API_KEY
OPENAI_API_KEY=<fallback; los scripts lo sobrescriben con LLM_API_KEY>
OPENAI_API_BASE_URL=http://localhost:11436/v1
```

**Alternativas de embeddings compatibles con MLX y oMLX** (sin descargar; la decisión es tuya):
- **Qwen3-Embedding-0.6B (conversión MLX), recomendada:** 1024 dimensiones, licencia Apache-2.0. oMLX la detecta de forma nativa (`Qwen3ForCausalLM` con «embedding» en el nombre de la carpeta, pooling `lasttoken`). Mantiene `EMBEDDING_DIMENSIONS=1024`.
- **BGE-M3 (conversión MLX):** 1024 dimensiones, licencia MIT, `XLMRobertaModel`, soporte nativo en oMLX (pooling `cls`) y multilingüe.
- **Mantener Jina v5:** habría que añadir `"architectures": ["Qwen3ForCausalLM"]` a su `config.json` (modificar el modelo instalado). No está verificado que oMLX reproduzca el pooling de su `model.py`, ni que aplique los prefijos, y la licencia es no comercial. No lo recomiendo.

**Para el NER:** probar otros modelos instalados (gemma-4-31B, Ornith) con `NERExtractor.extract`, o adaptar `ner_extractor.py` para aceptar una lista de entidades. Esto último es un cambio de código que requiere tu decisión.

## 10. Dictamen

**NO-GO para una validación real de extremo a extremo con la configuración actual**, por dos bloqueos verificados:
1. El modelo de embeddings configurado no lo sirve oMLX (400).
2. El LLM configurado no produce el formato de NER esperado (0 entidades).

Por los fallos silenciosos, la app no avisaría de ninguno de los dos.

**GO CON CONDICIONES para la capa de integración.** Código, configuración, Docker y Neo4j están verificados: 18/18 unitarias, 9/9 en Neo4j de laboratorio, chat real OK, conectividad Docker OK y la autenticación funciona en ambos clientes.

Condiciones para pasar a GO:
1. Instalar y configurar un modelo de embeddings que oMLX detecte como `embedding`, y fijar `EMBEDDING_DIMENSIONS` a su dimensión medida.
2. Corregir tu `.env`: `/v1` en `EMBEDDING_BASE_URL` y `EMBEDDING_DIMENSIONS`.
3. Elegir un LLM que pase `NERExtractor.extract`, o autorizar el cambio en `ner_extractor.py`.
4. Autorizar la regeneración de `uv.lock` con `requires-python = ">=3.11,<3.12"` (impacto: 0 cambios de versión).
5. Opcional, recomendado: hacer visibles los fallos de embeddings y NER en `add_text`.
