# Chatbot Promtior

Chatbot asistente que utiliza la arquitectura RAG (Retrieval Augmented Generation) para responder preguntas sobre el contenido del sitio web de Promtior, basado en la biblioteca LangChain.

## Live Demo

https://promtior.tatool.com.ar

## Características

- Procesamiento de URLs y documentos PDF
- Soporte para múltiples modelos:
  - Modelos API:
    - GPT-4
    - GPT-3.5 Turbo
  - Modelos Locales:
    - Llama 3.2
    - DeepSeek R1
- Interfaz web moderna y responsiva

## Requisitos

- Docker
- Claves de API:
  - OpenAI API Key (para modelos GPT)
  - LangChain API Key


## Principales desafíos encontrados

El primeer desafío fue comprender la arquitectura RAG y como implementarla en un proyecto, y adaptar mis conocimientos al framework de LangChain. 
Una vez superado este desafío, el siguiente se vio en elegir la mejor IA para el proyecto, como se puede ver en el live demo, se puede elegir entre GPT-4, GPT-3.5 Turbo, Llama 3.2 y DeepSeek R1, sin embargo, estas últimas dos solo están disponibles en el entorno de desarrollo, ya que no consegui forma de correrlas en producción.

El siguiente desafío me lo encontre a la hora de entender como funcionaban los splits, y su integración con los retrievers, me encontré con que la IA no era capaz de contestar a las preguntas a pesar de tener la informacion disponible, por ello implemente un sistema de logs y guarde los splits en un json para analizarlos mejor y encontrar la causa del problema. Al final lo solucioné modificando el tamaño de los splits, y el overlap de los mismos, ademas fue aqui donde introduje la IA DeepSeek R1, que resultó ser la mejor IA para el proyecto.

## Posibles mejoras

- Implementar un sistema que permita buscar en google para asi darle mas contexto a la IA
- Algunas webs como la de linkedin no se pueden scrapear, por lo que se debería implementar un sistema de scraping alternativo

## Configuración

1. Variables de Entorno:
```bash
LANGCHAIN_API_KEY=tu_clave_langchain
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=promtior-rag
OLLAMA_BASE_URL=http://127.0.0.1:11434
OPENAI_API_KEY=tu_clave_openai
ENVIRONMENT=development|production
```

2. Certificados SSL (Producción):
```bash
# Crear directorio para certificados
mkdir -p ssl-certs

# Copiar contenido de certificados
sudo cat /etc/letsencrypt/archive/[dominio]/fullchain1.pem > ssl-certs/fullchain.pem
sudo cat /etc/letsencrypt/archive/[dominio]/privkey1.pem > ssl-certs/privkey.pem

# Configurar permisos
sudo chmod 644 ssl-certs/*.pem
```

## Instalación

1. Clonar el repositorio:
```bash
git clone [url-repositorio]
cd Promtior-chatbot
```

2. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus claves
```

3. Iniciar servicios:
```bash
docker-compose up -d
```

## Uso

1. Acceder a la interfaz web:
   - Desarrollo: http://localhost:8000
   - Producción: https://[tu-dominio]

2. Funcionalidades:
   - Analizar páginas web usando @url
   - Subir documentos PDF
   - Seleccionar diferentes modelos de IA
   - Chatear con el asistente

## Estructura del Proyecto

```
Promtior-chatbot/
├── src/
│   ├── document_loader.py    # Carga y procesamiento de documentos
│   ├── processor.py          # Procesador principal RAG
│   ├── query_analyzer.py     # Análisis de consultas
│   ├── retriever.py         # Recuperación de documentos
│   └── prompts.py           # Plantillas de prompts
├── static/
│   └── index.html           # Interfaz web
├── logs/                    # Registros del sistema
├── splits/                  # Fragmentos de documentos procesados
├── ssl-certs/              # Certificados SSL
├── docker-compose.yml      # Configuración de Docker
└── nginx.conf              # Configuración de Nginx
```

## Seguridad

- Validación de API keys
- Restricción de modelos locales en producción
- Certificados SSL
- Headers de seguridad HTTP
- Logs de actividad

## Mantenimiento

- Los logs se encuentran en el directorio `logs/`
- Los certificados SSL deben renovarse manualmente
- Backups automáticos de fragmentos en `splits/`

## Entornos

### Desarrollo
- Permite todos los modelos
- Dependencia con ([Ollama](https://ollama.com))
- Logging detallado

### Producción
- Solo modelos API (GPT)
- Validación estricta de API keys

