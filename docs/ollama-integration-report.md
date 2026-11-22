# Informe técnico — Integración con Ollama (chat + embeddings)

Fecha: 2026-11-21 · Ollama 0.40.17 (nativo en macOS, `*:11434`) · Neo4j 5.18.1 (laboratorio aislado) · Python 3.11.15
Sustituye a `docs/omlx-integration-report.md`, que se conserva como histórico porque oMLX quedó descartado.
Restricciones respetadas: no se modificó `.env`, no se hizo commit ni push, no se tocaron datos ni volúmenes, y no se descargó ningún modelo.

**Recomendación: GO CON CONDICIONES.** Todas las pruebas ejecutadas pasan, incluido el build de grafo de extremo a extremo vía API. Las condiciones están en §6; ninguna es un bloqueo técnico.

> **Actualizaciones:**
> - **§7, Fase 2 (estabilización, 2026-11-22):** `uv.lock`, contexto, calidad de extracción, compose, simulación y reporte. Clasificación P0/P1/P2.
> - **§8, Fase 3 (`num_ctx` controlado desde la app):** evaluación de compatibilidad, mediciones y diseño propuesto. Pendiente de decisión.
> - **§9, Fase 4 (implementado):** `LLM_API_MODE=ollama_native`, `num_ctx` por operación, preflight sin truncamiento, modelo derivado para CAMEL y P1 del thinking. **Los 6 criterios de aceptación se cumplen con GGUF.**
> - La condición 2 de §6 («subir `OLLAMA_NUM_CTX`») queda **reemplazada** por §7.2 y §8.

---

## 1. Archivos modificados

| Archivo | Cambio | Motivo |
|---|---|---|
| `backend/app/config.py` | Valores por defecto `EMBEDDING_MODEL=qwen3-embedding:0.6b` y `EMBEDDING_DIMENSIONS=1024`, ambos con base `localhost:11434/v1`. Se mantienen `EMBEDDING_API_KEY` y `EMBEDDING_DIMENSIONS` configurables. | Ollama como proveedor por defecto, con valores coherentes entre sí. |
| `backend/app/utils/llm_client.py` | `chat_json(schema=…)` envía un `response_format` de tipo `json_schema`. `disable_thinking` envía `reasoning_effort:"none"` (solo a URLs de Ollama). Error explícito si la respuesta se trunca (`finish_reason=length`). `content=None` se trata como `""`. Los valores por defecto no cambian para los llamadores existentes. | Causa raíz del NER vacío (§3) y diagnóstico claro de truncados. |
| `backend/app/storage/ner_extractor.py` | Contrato Pydantic (`NERResult`, `extra="forbid"`) y JSON Schema generado desde la ontología (tipos como `enum`). Reintentos acotados (`max_retries+1`) y después `NERExtractionError`. Ya no devuelve una lista vacía cuando falla. | Punto 3 del encargo. |
| `backend/app/storage/neo4j_storage.py` | `add_text` propaga los errores de embeddings (antes guardaba nodos sin vector). | Evitar grafos sin vectores de forma silenciosa. Falla antes de escribir nada del chunk. |
| `backend/app/api/graph.py` | Un build con 0 nodos marca la tarea como `FAILED` con un mensaje explicativo. | Evitar grafos vacíos silenciosos cuando el modelo no extrae nada. |
| `backend/app/services/ontology_generator.py` | `max_tokens` 4096 → 8192 y thinking desactivado. | El razonamiento consumía el presupuesto de tokens y truncaba el JSON de la ontología (reproducido). |
| `backend/app/storage/embedding_service.py`, `neo4j_schema.py` | Sin cambios en esta fase (las mejoras de la fase oMLX se conservan): formato OpenAI, validación de dimensión, errores claros, índices con la dimensión configurada. | — |
| `.env.example` | Ollama en el 11434 para chat y embeddings, variantes de Docker con `host.docker.internal`, nota de migración. | — |
| `docker-compose.yml` | Sin servicio `ollama` (corre nativo en macOS); `extra_hosts: host.docker.internal:host-gateway`. Validado con `docker compose config`. | Solo las dependencias necesarias. |
| `backend/tests/test_ner_extractor.py` (nuevo), `test_embedding_service.py` | 36 pruebas unitarias. | — |
| `CLAUDE.md` | Actualizado a Ollama. | — |

## 2. Pruebas ejecutadas

| ID | Prueba | Tipo | Resultado |
|---|---|---|---|
| U | Suite unitaria (`pytest tests`, 36 casos): embeddings, config, schema, contrato NER, opciones de `LLMClient`, propagación de errores en `add_text` | Mocks | **PASS** 36/36 |
| C1 | Ollama `/api/version` desde el host | Real | **PASS** (0.40.17) |
| C2 | Desde un contenedor: `host.docker.internal:11434` (con `host-gateway`) | Real | **PASS**: versión y embedding de 1024 |
| C3 | Desde un contenedor: `localhost:11434` | Real | Rechazado, como se esperaba (es el propio contenedor) |
| E0 | `/api/show qwen3-embedding:0.6b` | Real | **PASS**: `embedding_length=1024`, capacidad `embedding` |
| E1 | Embeddings reales con `EmbeddingService` | Real | **PASS**: 1024 dimensiones, norma 1,0 |
| E2 | `EMBEDDING_DIMENSIONS=768` con el modelo de 1024 | Real | **PASS**: rechazado con «returned 1024-dim vectors but EMBEDDING_DIMENSIONS=768» |
| J1 | `json_object` frente a `json_schema` (Qwen3.5:9b y qwen3.8:27b) | Real | Con `json_object`, qwen3.8 inventó `relationships`/`subject`. Con `json_schema` (`/v1`) y `format` (`/api/chat`), 4 de 4 cumplen el esquema. Confirma la causa raíz. |
| J2 | `reasoning_effort:"none"` en 3 modelos | Real | **PASS**: 8,9 s → 1,4 s; inocuo en modelos sin thinking |
| N1 | `NERExtractor.extract` real (qwen3.8:27b-mxfp8), texto en español | Real | **PASS**: 5 entidades (tipos de la ontología, incluido `MediaOutlet`), 3 relaciones, 6,9–16,8 s |
| A0 | `create_app` con el Neo4j de laboratorio | Real | **PASS** |
| A1 | API: `ontology/generate` → `build` → tarea `COMPLETED` | Real, extremo a extremo | **PASS**: 10 tipos de ontología, 5 nodos, 4 aristas, 53 s (primer intento FAIL por truncado de la ontología; corregido, §3) |
| A2 | Todos los nodos y relaciones con vector de 1024 | Real | **PASS**: 5/5 entidades, 4/4 relaciones |
| A3 | Búsqueda híbrida (vector + BM25) | Real | **PASS**: 4 hechos. Con `scope="both"`, el nodo top es «Banco Central…» con puntuación 1,0 |
| F1 | Build con modelo de embeddings inexistente | Real | **PASS**: tarea `FAILED`, mensaje «Embedding request failed (404): model … not found» |
| R1 | `compileall` de `app/` y `scripts/` | Regresión | **PASS** |
| R2 | `docker compose config` | Regresión | **PASS** |
| B1 | `docker compose up` completo (app en contenedor) | — | **BLOCKED**: no ejecutado; solo se probó la red del contenedor (C2) |
| B2 | Simulación OASIS y generación de reporte de extremo a extremo | — | **BLOCKED**: no ejecutado (fuera del alcance del grafo). Usan el mismo `LLMClient` y `LLM_*`; no se tocaron |
| B3 | Host Linux con `host-gateway` | — | **BLOCKED**: no hay host Linux disponible |

Comandos:
- Unitarias: `cd backend && <venv 3.11>/bin/python -m pytest tests -q`
- Extremo a extremo: `scratchpad/ollama_e2e_test.py <modelo>`, que levanta un Neo4j desechable en 127.0.0.1:17687, sin volumen, eliminado al terminar. La configuración se sobrescribe en el proceso y los proyectos se escriben en el scratchpad; `backend/uploads` no se tocó.

## 3. Problemas encontrados y corregidos

1. **NER vacío (causa raíz del fallo visto con oMLX).** El modo `json_object` solo garantiza JSON válido, no la forma. Los modelos devuelven `relationships` o `subject`/`object`, o una lista, y el parser lo descartaba en silencio. Se corrigió con un JSON Schema forzado por Ollama, validación Pydantic y una excepción si tras los reintentos no hay respuesta válida.
2. **Ontología truncada.** Con thinking activo, el razonamiento más un JSON de unos 12.000 caracteres superaban `max_tokens=4096`; el error aparecía como «Invalid JSON». Se corrigió desactivando el thinking, subiendo a 8192 y detectando el truncado explícitamente.
3. **Grafos sin vectores o vacíos en silencio.** `add_text` tragaba los errores de embeddings y NER, y el build terminaba `COMPLETED` con 0 nodos. Ahora el error se propaga y un build con 0 nodos marca `FAILED`.

## 4. Errores y riesgos pendientes

1. **Tu `.env` sigue apuntando a oMLX** (`10.0.0.13:11436`, modelos de oMLX, `EMBEDDING_BASE_URL` sin `/v1`, sin `EMBEDDING_DIMENSIONS`). No lo modifiqué. Valores a poner:
   ```env
   LLM_API_KEY=ollama
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_MODEL_NAME=qwen3.8:27b-mxfp8
   EMBEDDING_MODEL=qwen3-embedding:0.6b
   EMBEDDING_BASE_URL=http://localhost:11434/v1
   EMBEDDING_DIMENSIONS=1024
   OPENAI_API_KEY=ollama
   OPENAI_API_BASE_URL=http://localhost:11434/v1
   ```
2. **`OLLAMA_NUM_CTX=8192` (por defecto) limita prompt y salida juntos.** La ontología admite documentos de hasta 50.000 caracteres (unos 12.000 tokens), que no caben, y Ollama trunca la entrada. Para documentos largos, sube `OLLAMA_NUM_CTX` (por ejemplo a 32768; el modelo lo soporta, a cambio de más memoria). No se probó con documentos largos.
3. **Builds parciales.** Si falla el chunk N, los chunks anteriores ya están escritos. La tarea queda `FAILED` y visible, pero el grafo parcial permanece (es el comportamiento previo del builder).
4. **`uv.lock` desincronizado** (anterior a este trabajo): `uv run`/`uv sync` lo reescriben. Regenerarlo con `requires-python = ">=3.11,<3.12"` no cambia ninguna versión (analizado). No se aplicó: requiere tu autorización. Python 3.11 es un requisito real (`camel-oasis <3.12`). No se añadieron dependencias; Pydantic ya estaba declarado.
5. **Calidad de recuperación.** Qwen3-Embedding recomienda una instrucción en las consultas. La app no la añade: funciona (búsqueda correcta en A3), pero no se optimizó.
6. **Exposición de red.** Ollama escucha en `*:11434`, sin autenticación, y es accesible desde la LAN. Si no lo necesitas, enlázalo a 127.0.0.1 con `OLLAMA_HOST`; Docker Desktop seguiría llegando por `host.docker.internal`. No se cambió (configuración global).
7. **Tiempos.** Ontología unos 35–40 s y unos 7–17 s por chunk de NER con qwen3.8:27b. Documentos grandes tardarán proporcionalmente.

## 5. Neo4j y dimensión

- Los índices se crean con `EMBEDDING_DIMENSIONS` (1024 por defecto) y similitud coseno. Los vectores de qwen3-embedding vienen normalizados, así que son compatibles.
- En esta máquina **no existe ninguna base de Neo4j ni volumen de MiroFish**, así que no hay embeddings previos que mezclar ni migrar.
- Si en otra instalación existen índices de 768 (nomic-embed-text), `IF NOT EXISTS` no los actualiza, y una consulta de 1024 sobre un índice de 768 falla (verificado en laboratorio). Hay que eliminar `entity_embedding`/`fact_embedding`, reiniciar el backend y regenerar los vectores. El procedimiento completo, con respaldo y rollback, está en `docs/omlx-integration-report.md` §8; es independiente del proveedor. Requiere autorización: es destructivo.

## 6. Recomendación

**GO CON CONDICIONES.** El flujo ontología → build → vectores → búsqueda funciona de extremo a extremo con Ollama, y los fallos ahora son visibles en lugar de silenciosos.

Condiciones:
1. Actualizar tu `.env` con los valores de §4.1.
2. Para documentos largos, subir `OLLAMA_NUM_CTX`.
3. Antes de producción, una validación con `docker compose up` completo y una simulación y reporte reales (B1 y B2).
4. Opcional: autorizar la regeneración de `uv.lock` (sin cambios de versión).

---

## 7. Fase 2 — Estabilización (2026-11-22)

Modelo de las pruebas: `qwen3.8:27b-mxfp8` (MLX). Embeddings: `qwen3-embedding:0.6b`. Neo4j en laboratorios desechables. El corpus es sintético, con nombres ficticios y verdad de referencia (`scratchpad/corpus.py`).

### 7.1 Dependencias (autorizado y aplicado)
- `pyproject.toml`: `requires-python = ">=3.11,<3.12"`, porque `camel-oasis 0.2.5` exige `<3.12`. Nuevo `backend/.python-version` = `3.11`.
- `uv.lock` regenerado con **0 cambios de versión** en 183 paquetes. Se eliminan `zep-cloud` (resto del proyecto original), `setuptools` y la raíz `mirofish-backend` 0.1.0, que pasa a `mirofish-offline-backend` 0.2.0; `neo4j` pasa a ser dependencia directa. El diff de −998 líneas corresponde sobre todo a las ramas de resolución para Python ≥3.12, que desaparecen.
- **Reproducibilidad:** `uv lock --check` pasa, un re-lock es idéntico byte a byte, `uv sync --locked` funciona en un entorno limpio, y la imagen Docker instala 130 paquetes desde el lock.
- **Causa de `tiktoken`:** sin límite superior, `uv` elegía Python 3.15, y `tiktoken 0.7.0` solo publica wheels hasta cp312, así que intentaba compilar con Rust.

### 7.2 Contexto y documentos largos (medido)
| Documento | Caracteres | Tokens de entrada (ontología) | Tokens de salida | `finish_reason` | Tiempo |
|---|---|---|---|---|---|
| S | 1.651 | 1.765 | 2.249–2.397 | stop | 36–52 s |
| M | 7.614 | 3.019 | 2.432–2.744 | stop | 42–51 s |
| L | 48.605 (tope de la app: 50.000) | 11.151 | 3.301–3.554 | stop | 59–95 s |

- **Sin truncamiento en ningún caso:** los tokens de entrada son idénticos con `OLLAMA_NUM_CTX` en 8192 y en 32768.
- **El contexto real lo fija tu servidor:** `OLLAMA_CONTEXT_LENGTH=131072` en `~/Library/LaunchAgents/com.user.ollama-env.plist`. El `OLLAMA_NUM_CTX` de la app **no tiene efecto** (ver §8).
- **No hace falta segmentar:** el NER trabaja con chunks de ~500 caracteres y la ontología está topada en 50.000 (~11.000 tokens de entrada).
- **Memoria:** con ese contexto, qwen3.8 ocupaba 30–35 GB de 64 durante la medición, y 43 GB con varias peticiones en paralelo.

### 7.3 Calidad de extracción (build completo vía API)
| Documento | Chunks | Nodos | Aristas | Cobertura personas / organizaciones / relaciones | Duplicados de entidad | Fuera de ontología | No respaldadas | Build |
|---|---|---|---|---|---|---|---|---|
| S | 5 | 9 | 13–23 | 4/4 · 3/3 · 5/5 | 0 | 0 | 0 | 40–52 s |
| M | 23 | 30 | 73–81 | 18/18 · 10/10 · 23/23 | 0 | 0 | 0 | 166–174 s |
| M (rebuild `force`) | 23 | 30 | 68 | 18/18 · 10/10 · 23/23 | 0 | 0 | 0 | 142 s (Jaccard de entidades = 1,0) |
| L | 159 | 61 | 70 | 45/45 · 16/16 · 53/53 | 0 | 0 | 0 | 253 s |
| X (receta, sin entidades) | 2 | 0 | 0 | — | — | — | — | `COMPLETED` con `result.warning` |

- **Relaciones duplicadas (corregido).** Se creaban con `CREATE`: 29 de 81 en M estaban duplicadas, y 30 de 57 en total tenían el texto del hecho idéntico. Ahora un `MERGE` por (origen, destino, tipo, hecho) reutiliza la arista y añade el episodio. Tras el cambio, M queda en **3** duplicados con la misma cobertura; los restantes son paráfrasis.
- **Variabilidad del LLM:** con temperatura 0,1, S dio 13 aristas en una ejecución y 23 en otra.
- **Vacío legítimo frente a error técnico.** Un error de NER o embeddings deja la tarea en `FAILED` con la causa. Si todos los chunks se procesan bien y no hay coincidencias, termina `COMPLETED` con aviso (cambiado respecto a la Fase 1, a petición tuya).

### 7.4 Pruebas de despliegue, simulación, reporte y recuperación
| Prueba | Resultado |
|---|---|
| `docker compose up` aislado (proyecto `mirofish-lab`, 127.0.0.1, `env_file: !override`, uploads en el scratchpad) | **PASS**: frontend 200, backend 200, Neo4j `healthy`. El error `xdg-open ENOENT` de Vite es inocuo (P2) |
| Build del grafo dentro del contenedor | **PASS** (11 nodos, 25 aristas) |
| Simulación con el flujo de la interfaz (Twitter y Reddit, `parallel`, 10 rondas) | **PASS**: `completed`, 59 acciones (21 + 38), preparación 602 s, ejecución 676 s |
| Simulación con `platform=twitter`, 2 rondas (solo por API) | Sin acciones: las rondas 0 y 1 caen a las 00:00–02:00 simuladas, sin agentes activos. Ese script tampoco escribe `actions.jsonl`, así que `run-status` nunca llega a `completed` (P2; la interfaz usa `parallel`) |
| Reporte | **PASS tras corregir un P0**: `completed`, 4 secciones, 28.574 caracteres, 0 caracteres chinos, **58 min** |
| Ollama inaccesible | Ontología: HTTP 500 «Connection error.». Build: `FAILED` «NER extraction failed after 2 attempts…: Connection error.» |
| Neo4j caído durante el build | `FAILED` en 5 s: «Cannot resolve address neo4j:7687» |
| Recuperación tras reiniciar Neo4j | **PASS**: un build nuevo termina `COMPLETED` sin reiniciar la app |
| Linux / `host-gateway` | **BLOCKED**: no hay entorno Linux autorizado |

### 7.5 Correcciones de esta fase
1. **P0, reporte imposible con Qwen en Ollama.** El `ReportAgent` pedía `<tool_call>` en texto plano. El parser nativo de Ollama para Qwen interceptaba ese token y devolvía HTTP 500 «EOF» (reproducido de forma aislada; `gemma4` no falla). Los prompts ahora piden `<tool>`, y el parser acepta ambas etiquetas.
2. **Seguridad, inyección de Cypher.** El tipo de entidad, generado por el LLM, se interpolaba como etiqueta. Ahora se escapan las comillas invertidas según la regla de Cypher.
3. Relaciones duplicadas (§7.3), extracción vacía legítima (§7.3) y lockfile (§7.1).
4. Pruebas unitarias: **40/40** (contrato NER, `LLMClient`, embeddings, config, schema y parser del reporte).

### 7.6 Clasificación de pendientes
| Prioridad | Pendiente | Evidencia |
|---|---|---|
| **P0** | Ninguno abierto | El P0 del reporte está corregido y verificado |
| **P1** | **Control de `num_ctx` desde la app**, ahora requisito obligatorio | §8: hoy no funciona con el modelo MLX ni por `/v1` |
| **P1** | Entrevistas del reporte superan el timeout IPC de 180 s | qwen3.8 con thinking tarda ~2,5 min por respuesta y hay 11 agentes con `OLLAMA_NUM_PARALLEL=4`, así que se encolan. El reporte termina sin esas entrevistas y tarda 58 min |
| P2 | Preparación lenta (10–15 min para 11 entidades) | Perfiles generados uno a uno con thinking |
| P2 | Paráfrasis duplicadas, y hechos a veces en inglés | §7.3 |
| P2 | El rebuild con `force` deja el grafo anterior huérfano en Neo4j | 2 grafos «M» en el laboratorio |
| P2 | `platform=twitter`/`reddit` nunca informa `completed` | §7.4 (solo por API; la interfaz no lo usa) |
| P2 | Error de Ollama inaccesible poco descriptivo («Connection error.») | §7.4 |
| P2 | Ollama expuesto en la LAN (`OLLAMA_HOST=0.0.0.0`, sin autenticación) | Configuración tuya |

---

## 8. Fase 3 — `num_ctx` controlado desde MiroFish (evaluación, sin cambiar modelo, `.env` ni servidor)

### 8.1 Compatibilidad en Ollama 0.40.17
- **Incidencia #18125** («mlxrunner: num_ctx … is not enforced»): **abierta**. Los PR de corrección #18261 y #18174 se **cerraron sin fusionar**: según el hilo, ignorar el contexto configurado en MLX es una decisión del proyecto. Ninguna nota de versión de 0.35.0 a 0.40.17 menciona un cambio.
- **Documentación oficial:**
  - La API compatible con OpenAI no permite fijar el contexto; propone un Modelfile.
  - La API nativa acepta `options.num_ctx`.
  - El código del runner MLX dice que «num_batch and num_ctx are not honored by this runner».
- **CAMEL 0.2.78** (agentes de OASIS): `OllamaModel` hereda de `OpenAICompatibleModel` (`/v1`) y `OllamaConfig` no tiene `num_ctx`, así que la simulación **no puede enviar `num_ctx`**.

### 8.2 Mediciones
Prompt de ~11.090 tokens. Memoria = diferencia en «In use system memory» del driver AGX (Metal real) respecto a la línea base sin modelos, que es distinta del tamaño de `ollama ps`.

| Modelo (runner) | `num_ctx` | Contexto en `ollama ps` | Prompt de 11k | `ollama ps` | Metal al cargar | Metal tras el prompt | NER S |
|---|---|---|---|---|---|---|---|
| `ornith-1.5:9b` (GGUF, llama.cpp) | 8192 | 8192 | **Rechazado, HTTP 400** «request (11090 tokens) exceeds the available context size (8192 tokens)» | 5,61 GB | **7,85 GB** | 7,85 GB | 7/7 |
| | 16384 | 16384 | Aceptado | 6,50 GB | 8,44 GB | 8,45 GB | 7/7 |
| | 32768 | 32768 | Aceptado | 7,98 GB | **9,65 GB** | 10,67 GB | 7/7 |
| `qwen3.8:27b-dflash2-nvfp4` (MLX, tu modelo) | 8192 | 8192 | **Aceptado entero (límite ignorado)** | 19,76 GB | 19,71 GB | 20,43 GB | 7/7 |
| | 16384 | 16384 | Aceptado | 19,76 GB | 20,12 GB | 20,83 GB | 7/7 |
| | 32768 | 32768 | Aceptado | 19,76 GB | 19,97 GB | 20,58 GB | 7/7 |

Pruebas adicionales:
- **Una petición `/v1` (sin `num_ctx`) a un GGUF cargado con 8192 lo recarga con 131072:** 19,36 GB frente a 7,57 GB. Al volver a 8192, Metal se queda en 9,14 GB y no libera todo de inmediato.
- **`qwen2.5:3b` (GGUF de arquitectura antigua) con `num_ctx=512`:** recortó el prompt **en silencio**, 258 de 2.195 tokens. El rechazo explícito depende del motor de cada modelo.

**Conclusiones**
1. **MLX** (tu modelo): `num_ctx` **no se respeta**, ni por `/v1` ni por la API nativa, aunque `ollama ps` lo muestre. La memoria **no depende de `num_ctx`**: es el peso del modelo más la caché KV de los tokens **realmente procesados** (~0,7 GB por 11.000 tokens; se multiplica por las peticiones en paralelo). En MLX, la única palanca es **limitar desde la app los tokens de entrada y salida**, que acota la caché KV real pero no el peso del modelo.
2. **GGUF / llama.cpp:** `num_ctx` **se respeta**, con rechazo explícito en el motor nuevo, y reduce memoria de forma medible: −1,8 GB (−19 %) de 32768 a 8192 en 9B, y **−11,8 GB (−61 %)** respecto al contexto por defecto de tu servidor (131072). Cambiar `num_ctx` recarga el modelo (~1,2 s en 9B).
3. Mezclar en el **mismo** modelo GGUF llamadas nativas con `num_ctx` y llamadas `/v1` (CAMEL) provoca **recargas continuas** y anula el ahorro.

### 8.3 Modificaciones propuestas (no implementadas; requieren tu aprobación)
1. **Adaptador nativo en `LLMClient`**, activado por `LLM_API_MODE=ollama_native`. El valor por defecto, `openai`, deja el comportamiento actual intacto. El modo nativo usa `POST /api/chat` con:
   - `options.num_ctx` y `options.num_predict`;
   - `format` = JSON Schema (se mantiene la validación Pydantic de `NERResult` y `_validate_and_process`);
   - `think` según la operación;
   - los mismos errores y timeouts.
   Cubre ontología, NER, perfiles, configuración de la simulación y reporte; todos pasan por `LLMClient`.
2. **`num_ctx` por operación desde el entorno:** `OLLAMA_NUM_CTX_DEFAULT` más `OLLAMA_NUM_CTX_{ONTOLOGY,NER,PROFILE,SIM_CONFIG,REPORT,CHAT}`, que recurren al valor por defecto si no se definen. Se recomiendan **pocos valores distintos**, para no recargar en cada cambio. Cada llamada recibe `operation=…`.
3. **Política de contexto sin truncamiento silencioso:**
   - antes de enviar, una estimación conservadora de tokens (entrada + `num_predict` ≤ `num_ctx`);
   - si no cabe, `ContextOverflowError`: rechazo explícito, o segmentación donde tenga sentido (ontología: `ONTOLOGY_OVERFLOW_POLICY=reject|segment`; el NER ya está segmentado);
   - después de responder, se mapea el HTTP 400 de Ollama a `ContextOverflowError`, y se detecta el truncamiento silencioso de motores antiguos (`prompt_eval_count` casi igual a `num_ctx`).
   - En MLX esta política es la **única** aplicación real del límite.
4. **Simulación (CAMEL):** no puede enviar `num_ctx`. Con GGUF, la opción viable es un **modelo derivado con `PARAMETER num_ctx`** (`ollama create`, local y sin descargas, pero añade un modelo a tu Ollama; requiere autorización). Así las llamadas `/v1` de los agentes usan ese contexto en lugar de 131072. Sin esto, la simulación no queda acotada y provoca recargas.
5. Eliminar el `extra_body.options.num_ctx` actual del modo `/v1`, que no hace nada, o convertirlo en alias de `OLLAMA_NUM_CTX_DEFAULT`.

### 8.4 Decisiones necesarias
1. **Modelo.** El criterio de aceptación (respetar `num_ctx` y reducir memoria) **solo se cumple con GGUF**. Tus opciones instaladas son:
   - `ornith-1.5:9b` (qwen35, Q4_K_M): NER 7/7 en S; falta validar ontología, M/L, simulación y reporte;
   - `bonsai-2:27b` (Q2_0, cuantización agresiva);
   - `flint:40b-a2b`.
   La alternativa es autorizar la descarga de un GGUF de Qwen3.x. Con tu modelo MLX actual, el adaptador solo aportaría los límites aplicados por la app.
2. **Autorizar la implementación** de §8.3, puntos 1–3 y 5.
3. **Autorizar `ollama create`** para el modelo derivado de la simulación (§8.3, punto 4), o aceptar que la simulación queda sin control de contexto.

> Decisión (2026-11-22): autorizados los puntos 1–3, y el 4 de forma condicional. Implementación y resultados en §9.

---

## 9. Fase 4 — `num_ctx` por operación, CAMEL y P1 del thinking (implementado y validado)

Sin cambios en tu `.env` ni en la configuración global de Ollama, sin descargas ni borrado de modelos, y sin commits. Laboratorio: compose aislado `mirofish-lab` (ya eliminado). Modelo GGUF: `ornith-1.5:9b`.

### 9.1 Cambios de código
| Archivo | Cambio |
|---|---|
| `app/config.py` | `LLM_API_MODE` (`openai` por defecto, o `ollama_native`); `OLLAMA_NUM_CTX_DEFAULT` y `OLLAMA_NUM_CTX_{ONTOLOGY,NER,PROFILE,SIM_CONFIG,REPORT,CHAT}` (el legado `OLLAMA_NUM_CTX` se acepta como valor por defecto); `CONTEXT_CHARS_PER_TOKEN` (3,5, medido); `SIMULATION_LLM_MODEL_NAME`; `SIMULATION_DISABLE_THINKING`; validación de `LLM_API_MODE` |
| `app/utils/llm_client.py` | Punto de entrada común `complete()` con dos transportes. **Nativo:** `POST /api/chat` con `options.num_ctx` y `num_predict`, `format` (JSON Schema o `"json"`) y `think`. **Preflight:** sistema + historial + prompt + esquema + reserva de salida ≤ `num_ctx`; si no cabe, `ContextOverflowError`. El HTTP 400 de contexto de Ollama se convierte en `ContextOverflowError`, y una ventana llena se trata como truncamiento sospechoso. Reintentos acotados ante errores de conexión o 5xx; los 4xx fallan sin reintentar. Registro `[llm]` por llamada (operación, `num_ctx`, estimación, tokens reales). Se elimina el `extra_body.num_ctx` del modo `/v1`, que no tenía efecto |
| `ner_extractor.py`, `ontology_generator.py`, `report_agent.py`, `graph_tools.py` | Operación explícita en cada llamada (el chat del reporte usa `chat`) |
| `oasis_profile_generator.py`, `simulation_config_generator.py` | Pasan de un cliente OpenAI propio a `LLMClient.complete()`. `ContextOverflowError` atraviesa los respaldos por reglas y los hilos, en lugar de ocultarse |
| `scripts/run_{parallel,twitter,reddit}_simulation.py` | Modelo `SIMULATION_LLM_MODEL_NAME` (con `LLM_MODEL_NAME` de respaldo); `model_config_dict={"reasoning_effort": "none"}` si `SIMULATION_DISABLE_THINKING=true`. **Sin tocar código de CAMEL** |
| `report_agent.py`, `graph_tools.py`, generadores | Correcciones descubiertas en la validación (§9.5): las respuestas vacías se reintentan; reserva de 8192 tokens en las secciones; thinking desactivado solo en salidas JSON estructuradas (perfiles, configuración de la simulación, selección de agentes) y en el resumen de entrevistas, este último con reserva de 2048 |
| `tests/test_llm_native.py` (nuevo) | 16 pruebas: payload nativo, `num_ctx` por operación, preflight, 400, 4xx, reintentos, ventana llena, el modo `openai` no envía `num_ctx`, y la propagación del error en NER, perfiles y configuración. **Total: 56/56** |

### 9.2 Modelo derivado para CAMEL: condiciones de autorización
| Condición | Verificación | Resultado |
|---|---|---|
| 1. CAMEL lo usa por un nombre configurable | `ModelFactory.create(OPENAI, model_type=<nombre>)` con CAMEL 0.2.78 real: la petición sale con `model="ornith-1.5:9b-mirofish-ctx32k"`; los scripts leen `SIMULATION_LLM_MODEL_NAME` | **Cumple** |
| 2. Mismo modelo salvo el contexto | `diff` de `ollama show --modelfile`: única diferencia, `PARAMETER num_ctx 32768`; mismos blobs; el original no define ningún `PARAMETER` | **Cumple** |
| 3. Respeta el contexto con CAMEL | `/v1` sin `num_ctx` carga 32768; un prompt de 39.642 tokens recibe HTTP 400 «exceeds the available context size (32768 tokens)»; en el flujo completo, los agentes CAMEL siempre a 32768 | **Cumple** |
| 4. Ninguna petición de la simulación impone 131072 | 3 ejecuciones completas con muestreo cada 2 s (532–789 muestras): el derivado solo aparece con **16384 y 32768**, nunca con 131072. Simulación y reporte, **sin recargas** | **Cumple** |
| 5. Sin modificar CAMEL ni mantener un fork | Solo el nombre del modelo y `model_config_dict` (campo de `ChatGPTConfig` que CAMEL reenvía tal cual) en nuestros scripts | **Cumple** |

Creado: `ollama create ornith-1.5:9b-mirofish-ctx32k -f Modelfile` (`FROM ornith-1.5:9b` + `PARAMETER num_ctx 32768`). Rollback: `ollama rm ornith-1.5:9b-mirofish-ctx32k`; el original queda intacto.

### 9.3 Contexto efectivo, recargas y memoria (nivel Ollama, derivado)
| Paso | Contexto | Memoria Metal (delta) | Tiempo |
|---|---|---|---|
| `/v1` sin `num_ctx` (como CAMEL) | 32768 | 9,90 GB | 1,3 s (carga) |
| `/v1` consecutiva, y nativa con 32768 | 32768, sin recarga | 9,90 GB | 0,2 s |
| `/v1` con 39.642 tokens | HTTP 400 | — | 0,1 s |
| Concurrentes: 4 por `/v1` + 1 nativa con 32768 | 32768, sin recarga, 5×200 | 9,94 GB | 1,9 s |
| Nativa con 16384 (perfiles) | 16384 (recarga) | 8,46 GB | 1,3 s |
| `/v1` después | 32768 (recarga) | 10,2 GB | 1,3 s |
| Referencia: el mismo GGUF por `/v1` sin derivado | **131072** | **19,36 GB** | — |

**Reducción medible: de 19,36 a ~9,9 GB (−49 %)** al pasar de 131072 a 32768, y −1,5 GB adicionales con 16384. En el flujo completo con base limpia (ejecución 1), el máximo por fase fue de 9,7 a 13,6 GB, incluidos los ~3 GB del modelo de embeddings. Con qwen3.8 MLX eran 30–43 GB.

### 9.4 Flujo completo (compose, `ollama_native`, derivado, 3 ejecuciones)
| Fase | Ejecución 1 | Ejecución 2 | Ejecución 3 |
|---|---|---|---|
| Ontología | OK, 10 tipos, 40 s | OK, 33 s | OK, 43 s |
| Grafo | 9 nodos / 13 aristas, 27 s | 9 / 12, 21 s | 10 / 25, 39 s |
| Preparación (perfiles 16384, configuración 32768) | 311 s, 2 perfiles truncados | **100 s, 0 truncados** | 145 s, 0 |
| Simulación CAMEL (`parallel`, 10 rondas, actualización del grafo) | 53 acciones, 320 s | 28, 130 s | 37, 200 s |
| Reporte | 901 s, **sección 1 vacía** | 791 s, 3 secciones con contenido | 811 s, 3 con contenido, 17.915 caracteres |
| Chat del reporte | OK | OK (2.003 caracteres) | OK (1.777 caracteres) |
| Entrevistas | 3, **0 timeouts** | 3, 0 | 3, 0 |
| `ContextOverflowError` / perfiles de respaldo / configuraciones por reglas | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Contextos vistos del derivado | 16384, 32768 | 16384, 32768 | 16384, 32768 |

**Tokens reales por operación** (máximo en las ejecuciones 2–3, frente a `num_ctx`):

| Operación | `num_ctx` | Entrada máx. | Salida máx. | Estimación del preflight (entrada) |
|---|---|---|---|---|
| ontología | 32768 | 1.763 (S; ~11.200 en L, §7.2) | 2.849 | 2.297 |
| NER | 32768 | 1.013 | 635 | 1.581 |
| perfiles | 16384 | 1.170 | 1.636 | 1.520 |
| configuración de la simulación | 32768 | 1.257 | 1.540 | 1.450 |
| reporte | 32768 | **15.622** | 4.096 (selección de agentes, corregida) | 22.204 |
| chat | 32768 | 3.864 | 490 | 5.712 |

- El reporte, la única operación que se acerca al límite, usa el ~48 % de 32768 en entrada. **No conviene bajarlo a 16384.**
- La ratio real fue de 4,0 a 5,5 caracteres por token, así que el preflight con 3,5 sobrestima entre un 15 y un 60 %: margen de seguridad sin rechazos en falso.

### 9.5 Problemas encontrados en la validación y corregidos
1. **Sección del reporte vacía, sin aviso.** Con el thinking activo, el modelo gastó los 4096 tokens de `num_predict` en razonar y devolvió `""`; el bucle ReAct solo reintentaba con `None` y adoptó el vacío como sección. **Corrección:** una respuesta vacía se trata como `None` (reintento), la reserva de las secciones pasa a 8192, y la conclusión forzada vacía produce el aviso visible existente. Verificado: las 3 secciones con contenido en las ejecuciones 2 y 3.
2. **Perfiles truncados** (2 de 7) con el thinking activo y 4096 tokens. **Corrección:** thinking desactivado en perfiles y configuración, que son JSON estructurado. Verificado: 0 truncados, y la preparación baja de 311 a 100–145 s.
3. **Resumen de entrevistas cortado** (3 de 3) a 800 tokens, cuando el propio prompt permite ~1.300. **Corrección:** reserva de 2048 tokens y sin thinking. Verificado: 0 cortes en la ejecución 3.
4. **Selección de agentes para entrevistar truncada** (3 de 3, 60 s cada una) y sustituida por la selección por defecto. **Corrección:** sin thinking en esa llamada. Verificado directamente: 6–8 s, ~470 tokens, `finish=stop`, selección hecha por el LLM.

### 9.6 Manejo de errores (real, contra Ollama)
| Caso | Resultado |
|---|---|
| Preflight: ~64.800 tokens estimados + 4096 > 16384 (perfiles) | `ContextOverflowError` «…Increase OLLAMA_NUM_CTX_PROFILE or reduce the input; it is not truncated.» y **0 peticiones enviadas** a Ollama |
| Servidor (preflight desactivado): 39.639 tokens con 32768 | `ContextOverflowError` con el mensaje 400 de Ollama |
| Modelo inexistente | `RuntimeError` 404, sin reintentos |
| Ollama inaccesible | `RuntimeError` tras 2 intentos acotados |
| Propagación al estado de la tarea | NER → `NERExtractionError` → `add_text` → build `FAILED` (mismo camino verificado en §7.4). Perfiles y configuración: la preparación falla en lugar de usar respaldos (pruebas unitarias) |

### 9.7 P1: thinking y timeouts de las entrevistas
- **Causa.** La entrevista por lotes lanza una petición por agente a la vez (11 en la prueba) y Ollama atiende 4 en paralelo (`OLLAMA_NUM_PARALLEL=4`). Con un modelo lento que piensa, las últimas superan los 180 s del IPC.
- **Con `ornith` GGUF y el thinking activo:** 9 entrevistas en 3 ejecuciones, **0 timeouts**.
- **Con tu `qwen3.8:27b-dflash2-nvfp4` (MLX)**, 11 entrevistas concurrentes:
  - thinking activo: la más lenta, 91,4 s (mediana 62,7 s, 427 tokens);
  - `reasoning_effort: "none"`: la más lenta, **32,4 s (−65 %)** (mediana 22,2 s, 124 tokens).
  Las entrevistas reales llevan el historial del agente: con qwen3.8 llegaron a ~2,5 min por respuesta (§7.6).
- **Aplicado:** `SIMULATION_DISABLE_THINKING` solo afecta a los agentes de la simulación. **Recomendado `true` con modelos lentos que piensan** (qwen3.8); con `ornith` no hace falta. El thinking se conserva en las secciones del reporte y en el chat. Los timeouts no se han tocado.

### 9.8 Criterios de aceptación
| # | Criterio | Estado | Evidencia |
|---|---|---|---|
| 1 | `num_ctx` controlado desde MiroFish y respetado en GGUF | **Cumplido** | §9.3 y §8.2: 16384/32768 aplicados, 400 por encima del límite; logs `[llm]` por operación |
| 2 | Simulación CAMEL sin recargas a 131072 | **Cumplido** | §9.2 (condición 4): 0 apariciones de 131072 en 3 ejecuciones; sin recargas en la simulación |
| 3 | Reducción medible de memoria | **Cumplido** | 19,36 → 9,9 GB (−49 %); flujo completo 9,7–13,6 GB frente a 30–43 GB con qwen3.8 |
| 4 | Ontología, entidades, grafos, simulación y reportes funcionales | **Cumplido** | §9.4, 3 ejecuciones |
| 5 | Entrevistas sin timeouts sistemáticos | **Cumplido** | 0 de 9 con `ornith`; con qwen3.8, `SIMULATION_DISABLE_THINKING=true` da un margen de 3× |
| 6 | Ninguna regresión P0/P1 | **Cumplido** | 56/56 unitarias; 4 regresiones detectadas en la validación y corregidas (§9.5); el modo `openai` sigue siendo el predeterminado e intacto |

**Limitación fundamental:** con modelos MLX (tu `qwen3.8:27b-dflash2-nvfp4`), `num_ctx` sigue sin poder respetarse, porque Ollama lo ignora por diseño (#18125). Ahí la app solo aplica sus propios límites (preflight y `num_predict`), y la memoria depende de los tokens realmente procesados.

### 9.9 Configuración para usar el modo GGUF
Para tu `.env`, que no he modificado. Las claves de esta configuración no son secretas.
```env
LLM_API_MODE=ollama_native
LLM_MODEL_NAME=ornith-1.5:9b-mirofish-ctx32k
SIMULATION_LLM_MODEL_NAME=ornith-1.5:9b-mirofish-ctx32k
OLLAMA_NUM_CTX_DEFAULT=32768
OLLAMA_NUM_CTX_PROFILE=16384
SIMULATION_DISABLE_THINKING=false   # true si vuelves a un modelo lento que piensa (qwen3.8)
# LLM_BASE_URL / OPENAI_API_BASE_URL siguen en http://localhost:11434/v1 (Docker: host.docker.internal)
```
Para volver a tu configuración actual basta con quitar esas líneas: `LLM_API_MODE` vuelve por defecto a `openai` con `qwen3.8:27b-dflash2-nvfp4`.

### 9.10 Pendientes (P2, sin nueva fase)
- Cada cambio entre 16384 (perfiles) y 32768 recarga el modelo (~1,3 s, 2 veces por preparación). Si se prefiere cero recargas, `OLLAMA_NUM_CTX_PROFILE=32768` cuesta ~1,5 GB.
- **P2 con condición de uso:** la calidad de `ornith-1.5:9b` en documentos M/L y en reportes largos se validó solo con el corpus S. En §7.3, con qwen3.8, M y L dieron un 100 % de cobertura. **Debe superarse esa evaluación antes de usar los resultados donde importe su exactitud** (ver §10.9).
- `chat()` registra un aviso cuando una respuesta de texto llega a `max_tokens`, pero la devuelve. `chat_json` lanza un error.
- El resto de P2 de §7.6 sigue igual.

## 10. Cierre de la estabilización Ollama + GGUF

Resultados aceptados por el responsable del proyecto. Dictamen: **GO TÉCNICO para desarrollo local** con Ollama y GGUF. Esta sección consolida lo necesario para reproducir y revertir; no añade ejecuciones ni cambios de arquitectura.

### 10.1 Resultados finales y criterios de aceptación
Los 6 criterios están cumplidos (tabla en §9.8): `num_ctx` aplicado por la API nativa y el runner GGUF; CAMEL con el modelo derivado, sin 131072; memoria −49 % (19,36 → 9,9 GB); pipeline completo 3/3; entrevistas sin timeouts (0 de 9 con ornith); 56/56 pruebas.

### 10.2 Decisiones finales
1. `ornith-1.5:9b` es el candidato principal para la integración GGUF.
2. `ornith-1.5:9b-mirofish-ctx32k` se conserva para CAMEL.
3. `LLM_API_MODE=ollama_native` es una opción explícita; el modo `openai` sigue siendo el predeterminado.
4. Contexto por defecto 32768; perfiles 16384. **No se cambian sin nueva evidencia.**
5. No se sube el contexto de perfiles solo para evitar las recargas de ~1,3 s.
6. `qwen3.8:27b-mxfp8` / `qwen3.8:27b-dflash2-nvfp4` (MLX) se conservan y siguen siendo utilizables.

### 10.3 Modelos y versiones
| Componente | Versión |
|---|---|
| Ollama | 0.40.17 (app de macOS, `ollama serve`, puerto 11434) |
| Chat / NER / agentes CAMEL (GGUF) | `ornith-1.5:9b` (id `e5df7dcdd8a2`, 6,6 GB), derivado `ornith-1.5:9b-mirofish-ctx32k` (id `575f4778cbfd`) |
| Embeddings | `qwen3-embedding:0.6b`, 1024 dimensiones |
| Alternativa conservada | `qwen3.8:27b-mxfp8`, `qwen3.8:27b-dflash2-nvfp4` (MLX) |
| Neo4j | 5.18-community (relationship vector search) |
| Python / CAMEL / OASIS | 3.11 exacto / camel-ai 0.2.78 / camel-oasis 0.2.5 |
| Hardware de las mediciones | Apple M4 Max, 64 GB |

### 10.4 Configuración reproducible de Ollama (global, no modificada por este trabajo)
Valores del servidor del responsable, tomados de su `~/Library/LaunchAgents/com.user.ollama-env.plist`: `OLLAMA_CONTEXT_LENGTH=131072`, `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=40m`, `OLLAMA_MAX_LOADED_MODELS=4`, caché KV `q8_0`. Con ellos, cualquier llamada `/v1` a un modelo sin `num_ctx` propio carga 131072; por eso el derivado.

**Modelfile exacto del modelo derivado** (`ollama show --modelfile` confirma que la única línea añadida al original es `PARAMETER num_ctx`):
```
FROM ornith-1.5:9b
PARAMETER num_ctx 32768
```
Creación: `ollama create ornith-1.5:9b-mirofish-ctx32k -f Modelfile`. Rollback: `ollama rm ornith-1.5:9b-mirofish-ctx32k`.

### 10.5 Variables de entorno (`.env`; ver `.env.example`)
| Variable | Valor recomendado (GGUF) | Notas |
|---|---|---|
| `LLM_API_MODE` | `ollama_native` | `openai` (por defecto) no envía `num_ctx` |
| `LLM_MODEL_NAME` | `ornith-1.5:9b-mirofish-ctx32k` | |
| `SIMULATION_LLM_MODEL_NAME` | `ornith-1.5:9b-mirofish-ctx32k` | CAMEL siempre usa `/v1`; el contexto lo fija el Modelfile |
| `OLLAMA_NUM_CTX_DEFAULT` | `32768` | |
| `OLLAMA_NUM_CTX_{ONTOLOGY,NER,SIM_CONFIG,REPORT,CHAT}` | `32768` | |
| `OLLAMA_NUM_CTX_PROFILE` | `16384` | |
| `SIMULATION_DISABLE_THINKING` | `false` con ornith; `true` con qwen3.8 | |
| `CONTEXT_CHARS_PER_TOKEN` | `3.5` | preflight; la ratio real es 4,0–5,5 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `qwen3-embedding:0.6b` / `1024` | cambiar la dimensión exige recrear los índices vectoriales |

Con la app en Docker, las URL usan `host.docker.internal`, no `localhost`.

### 10.6 Memoria: metodología y resultados
- **Medición:** la memoria GPU se leyó con `ioreg -c AGXAccelerator` (campo «In use system memory») y se restó una base medida con ningún modelo cargado (1,17 GB). El contexto efectivo y el runner salen de `ollama ps` (columnas CONTEXT y RUNNER), consultado en bucle cada 2 s durante los flujos completos (532–789 muestras por ejecución).
- **Resultados:** 19,36 GB con 131072 → 9,9 GB con 32768 (−49 %) → 8,46 GB con 16384. Flujo completo: 9,7–13,6 GB por fase (con embeddings), frente a 30–43 GB con qwen3.8 MLX. Detalle en §9.3.
- **Límite de la medición:** una sola máquina y un solo modelo GGUF; los valores absolutos no son transferibles, la proporción sí.

### 10.7 Correcciones de thinking, truncamiento y timeouts
- **Thinking** consume `num_predict`. Se desactiva (`think:false` en nativo, `reasoning_effort:"none"` en `/v1`) solo en salidas JSON estructuradas (NER, ontología, perfiles, configuración), en el resumen de entrevistas y en la selección de agentes. Se conserva en secciones del reporte y chat.
- **Truncamiento:** `finish_reason == "length"` en JSON lanza un error explícito; las secciones del reporte reservan 8192 tokens y reintentan ante respuesta vacía; el resumen de entrevistas reserva 2048. Nada se trunca en silencio (preflight + `ContextOverflowError`).
- **Timeouts:** no se han modificado. La causa de los 180 s era la cola de entrevistas concurrentes con un modelo lento que piensa; con `SIMULATION_DISABLE_THINKING=true` la más lenta pasó de 91,4 s a 32,4 s con qwen3.8 (§9.7).

### 10.8 Procedimiento manual para volver al modo OpenAI
1. En `.env`, borrar o comentar `LLM_API_MODE`, `SIMULATION_LLM_MODEL_NAME`, `SIMULATION_DISABLE_THINKING` y las `OLLAMA_NUM_CTX_*`. `LLM_API_MODE` vuelve a `openai`.
2. Restaurar `LLM_MODEL_NAME=qwen3.8:27b-dflash2-nvfp4` (o el modelo MLX que se use).
3. Reiniciar el backend (`npm run dev` o `docker compose restart`).
4. Opcional: `ollama rm ornith-1.5:9b-mirofish-ctx32k`; el original `ornith-1.5:9b` y los modelos qwen3.8 no se tocan.
Con MLX, `num_ctx` no se respeta (ollama/ollama#18125); la app solo aplica preflight y `num_predict`.

### 10.9 Limitaciones conocidas y pendientes P2
- **Calidad de `ornith-1.5:9b` con documentos grandes (P2, bloqueante para uso donde importe la exactitud):** solo se validó con el corpus S (documento pequeño). La evaluación con documentos M/L y reportes largos **debe superarse antes de considerar los resultados aptos para un uso donde importe su exactitud**. Hasta entonces, los resultados con ornith son válidos para desarrollo y pruebas funcionales.
- Cada cambio 16384 ↔ 32768 recarga el modelo (~1,3 s, 2 veces por preparación). Aceptado.
- MLX ignora `num_ctx`; CAMEL no puede enviar `num_ctx` por `/v1`.
- `chat()` de texto solo avisa si llega a `max_tokens`; `chat_json` falla.
- El resto de P2 de §7.6 sigue igual.
- No se abre ninguna nueva fase de investigación por P2.

### 10.10 Alcance y control de cambios
- **Dentro del proyecto:** los cambios son los archivos listados en §1, §7 y §9.1 (`git status` en el árbol de trabajo: `.env.example`, `backend/`, `docker-compose.yml`, `CLAUDE.md`, `docs/`), todos sin commit y sin push.
- **Fuera del proyecto, un único artefacto autorizado:** el modelo `ornith-1.5:9b-mirofish-ctx32k` (§10.4).
- **No modificados:** el `.env` del responsable, `ornith-1.5:9b`, los modelos qwen3.8 y de embeddings, la configuración global de Ollama (plist, variables del servidor). No se descargó ni eliminó ningún modelo. El laboratorio aislado `mirofish-lab` fue eliminado; la pila del responsable no se tocó.

### 10.11 Estado
**Fase de estabilización Ollama + GGUF: CERRADA.** Los pendientes P2 quedan registrados sin abrir otra fase. El siguiente trabajo corresponde a las funcionalidades principales de MiroFish, no a optimizaciones del proveedor LLM.
