# PhD_RAG
RAG chatbot for Emory University CS PhD students

# Setup

Please utilize poetry to manage dependencies. To install poetry, please follow the instructions [here](https://python-poetry.org/docs/#installation).
Once poetry is installed, please run the following command to install the dependencies:

```bash
poetry install
```
Once the dependencies are installed, run the poetry environment:

```bash
poetry env activate
```

please run the following command to start the chatbot:

```bash
uvicorn PhD_RAG.src.main:app --reload
```

Then go to 127.0.0.1/docs in your browser to interact with API endpoints.


The file structure is being followed as shown here:

https://github.com/zhanymkanov/fastapi-best-practices?tab=readme-ov-file#project-structure
