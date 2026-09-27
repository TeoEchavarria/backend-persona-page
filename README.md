# API de búsqueda del portafolio

Búsqueda semántica sobre las notas de proyectos y skills: devuelve los párrafos
exactos que responden a una consulta, con una banda de confianza honesta, un
contexto de sesión (Rocchio), recomendaciones por cadenas de Markov y una
estimación de la intención del visitante (HMM).

Este repo no tiene contenido. Las notas viven en el frontend
(`the-frontend-blueprint/src/content/notes/`), que las envía a
`PUT /admin/content`. La API las divide en fragmentos, calcula los embeddings
de lo nuevo o editado y guarda todo en Postgres.

```
api/
  app/
    main.py          app y rutas
    config.py        settings (variables de entorno o api/.env)
    db.py            conexión y consultas SQL
    schema.sql       tablas notes, chunks, intents, sessions, events
    embeddings.py    encoder local (sentence-transformers) o HTTP (Hugging Face)
    chunking.py      markdown -> fragmentos "Nota > Sección > párrafo"
    search.py        vectorial + texto, RRF, confianza
    session.py       Rocchio: effective_query, update_context
    visitor.py       estado de la sesión: contexto + intención
    markov.py        prior semántico, conteos, random walk, PageRank, MMR
    intent.py        HMM: algoritmo forward
    graph.py         grafo de notas e intenciones (caché de 60 s)
    ingest.py        sincroniza el contenido (solo recalcula lo que cambió)
    routes/          search, events, notes, admin
  scripts/
    calibrate.py     top-3 y umbrales de confianza con eval/queries.yaml
  eval/queries.yaml  48 consultas de prueba, 12 sin respuesta
  tests/             incluye unas notas de prueba en tests/fixtures
  index.py           punto de entrada para Vercel
```

## En local

```sh
docker compose up -d                 # Postgres 17 + pgvector en el puerto 5433
cd api
cp .env.example .env                 # define ADMIN_TOKEN
uv sync --extra local                # incluye sentence-transformers
uv run uvicorn app.main:app --reload --port 8000   # la primera vez descarga el modelo (~470 MB)

# desde el frontend, con SEARCH_ADMIN_TOKEN = ADMIN_TOKEN:
npm run search:sync

uv run pytest                        # usa la base portfolio_test (createdb aparte)
```

Las pruebas de endpoints necesitan una base `portfolio_test` en el mismo
servidor (`createdb -h localhost -p 5433 -U portfolio portfolio_test`) y usan un
encoder falso y determinista, así que no descargan el modelo. Si no hay
Postgres, se saltan y corren solo las de chunking y matemática.

## Endpoints

| Método | Ruta | Qué hace |
| --- | --- | --- |
| GET | `/health` | Verifica la conexión a la base |
| POST | `/search` | `{query, session_id?, kind?, since?, order?, use_context?}` → resultados (con el texto resaltado entre ⟦ ⟧), total de candidatos, tiempo, confianza, contexto e intenciones. `kind`: `project`, `skill` o `note`; `since`: año mínimo de publicación; `order`: `relevance` o `date`. |
| POST | `/events` | `{session_id, kind: read\|select\|finish, chunk_id?, note_slug?}` → estado de la sesión |
| GET | `/sessions/{id}` | Contexto e intenciones actuales |
| DELETE | `/sessions/{id}/context` | Quita el contexto |
| GET | `/notes` | Notas ordenadas por PageRank global |
| GET | `/notes/{slug}` | Nota completa, fragmento por fragmento |
| GET | `/notes/{slug}/related?k=3` | Random walk with restart + MMR |
| GET | `/graph` | Nodos, matriz de transición, prior y conteos, para el panel |
| PUT | `/admin/content` | `Authorization: Bearer ADMIN_TOKEN`. `{notes: [{slug, kind, source}], intents, force?}` reemplaza todo el contenido. Crea el esquema si la base está vacía. |

## Cómo decide

**Búsqueda híbrida.** Producto interno de pgvector (`<#>`) sobre vectores de
norma 1, que equivale a la similitud coseno, más `tsvector` en español con los
términos unidos por OR. Las dos listas (30 candidatos cada una) se fusionan con
`RRF(d) = Σ 1 / (60 + rango)` y se devuelven 5. Búsqueda exacta, sin índice
aproximado: con menos de ~50.000 fragmentos no hace falta.

**Confianza.** Se juzga con la consulta literal, no con la mezclada con el
contexto. Las bandas usan el **margen** (mejor similitud − similitud media de
todos los fragmentos) y la brecha entre el primero y el segundo. El z-score
también se calcula y se devuelve, pero no decide: en el set de prueba, las
consultas sin respuesta tienen una dispersión menor, lo que infla su z (2,58
para "horario de atención de la tienda", por encima de varias consultas con
respuesta). El margen separa mejor, pero no perfecto. Con 48 consultas (36 con
respuesta, 12 sin ella), el corte en 0,041 deja fuera de la banda lejana 35 de
36 consultas con respuesta, y dentro 9 de 12 sin respuesta. Las tres que se
escapan ("declaración de renta", "precio del dólar", "horario de atención")
caen en la banda media, cuyo mensaje ya advierte que puede no ser lo que se
busca. Se eligió ese error sobre el contrario: marcar como lejana una buena
respuesta. Entre 0,036 y 0,047 se mezclan ambas clases; contar palabras en
común no las separaba sin sobreajustar. El top-3 es 34/36.

Los umbrales dependen del corpus: al agregar notas, conviene volver a correr
`scripts.calibrate` y copiar los valores sugeridos al `.env`. Las consultas de
`eval/queries.yaml` apuntan a las notas actuales, así que hay que actualizarlas
junto con el contenido. Los embeddings de Hugging Face y los locales coinciden
(mismas similitudes a tres decimales), así que calibrar en local sirve para
producción.

**Resaltado.** Cada resultado trae `highlighted`, el párrafo con los términos
de la consulta marcados por `ts_headline` (con stemming en español, así que
"proyectos" resalta "proyecto"). Si el resultado llegó solo por significado,
puede no tener marcas.

**Contexto (Rocchio).** `read` y `select` actualizan `c ← βc + (1−β)e` con
β = 0,8, y la búsqueda vectorial usa `q_ef = αq + (1−α)c` con α = 0,7. El
contexto solo mueve la lista vectorial: si otro fragmento coincide también por
texto, RRF lo mantiene arriba. Es deliberado: el contexto desempata, no pisa
una coincidencia literal. Con el contexto de Tribe, "stack" y "cómo está
hecho" ponen a Tribe primero.

**Markov.** `P(j|i) = (c_ij + λ s_ij) / (Σ_k c_ik + λ)`, donde `s` es un
softmax de similitudes entre centroides de notas (τ = 0,015: con e5 todos los
centroides quedan entre 0,92 y 0,96 de coseno, así que un τ mayor da un prior
casi uniforme) y `c` son las
transiciones observadas en `events`. Salir de una nota leída hasta el final
cuenta doble. Sin eventos, `P = s`. Las relacionadas salen de un random walk
with restart (reinicio 0,3) y MMR (0,7). El orden de `/notes` es el PageRank
global (0,85).

**Intención (HMM).** Las intenciones (`intents.json` en el frontend) se representan
con el centroide de sus notas. Las emisiones son `exp(cos(o, μ)/τ)` y la
transición es pegajosa (0,8 en la diagonal). Cada consulta y cada lectura es
una observación, y el forward se actualiza en línea: la creencia vive en
`sessions.intent_belief`.

## Despliegue en Vercel

Vercel no puede cargar torch, así que el despliegue calcula el embedding de
cada consulta por HTTP con **el mismo modelo** con el que se indexó en local.
Por eso los vectores son compatibles.

1. **Base de datos.** Neon (Vercel Postgres): usa la cadena *pooled* con
   `?sslmode=require`. pgvector viene incluido, y la primera sincronización
   crea la extensión y las tablas.
2. **Proyecto en Vercel** con *Root Directory* = `api`. Vercel detecta FastAPI
   por `index.py` e instala las dependencias de `pyproject.toml` (sin el extra
   `local`, así que sin torch).
3. **Variables de entorno:**
   - `DATABASE_URL`
   - `ADMIN_TOKEN` (un secreto largo; el frontend lo usa como `SEARCH_ADMIN_TOKEN`)
   - `CORS_ORIGINS=https://www.echavarrias.com,https://echavarrias.com`
   - `EMBEDDING_BACKEND=http`
   - `EMBEDDING_API_TOKEN=hf_...` (token de lectura de Hugging Face)
   - opcional `EMBEDDING_API_URL`, si usas un Inference Endpoint dedicado; el
     formato de la petición es el mismo.
   - los umbrales `MARGIN_HIGH`, `MARGIN_MEDIUM` y `GAP_HIGH`, si los
     recalibraste.
4. **Contenido.** El build de producción del frontend corre
   `npm run search:sync` (postbuild) y envía las notas. Cada deploy con notas
   nuevas o editadas actualiza la base. Solo se calculan los párrafos que
   cambiaron, usando el mismo modelo por HTTP. Si cambias de modelo, usa
   `npm run search:sync -- --force`.

Memoria: el backend `http` no carga ningún modelo, así que una función de 1 GB
sobra. El backend `local` necesita ~1 GB por torch y el modelo (~470 MB).

## Privacidad

Los eventos guardan el id de sesión (un UUID aleatorio que el navegador guarda
en `sessionStorage`), el tipo de evento, el fragmento o nota y el texto de las
consultas. No se guardan IP ni user agent. Cada sincronización borra las
sesiones (y sus eventos) sin actividad en `EVENT_RETENTION_DAYS` días (180 por
defecto).
