# MediPal: Your AI Friend for Medical/Clinical Q&A

MediPal is an open-source medical assistant that provides comprehensive mediciation information and symptom-based recommendations using natural-language understanding which supports voice and message conversation. It is powered by local Huggingface LLMs, embedding model, cross-encoder(BERT) and whisper with agent frameworks like langchain, langgraph and many tools. 

---

## Table of Contents
- [Motivation](#motivation)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Tests](#tests)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Motivation
Recently I search for some medicine for my friend using online pharmacies, I noticed several gaps in the customer experience when I chat with their AI assistant.

1. No matter what I say, they just say what they have to say by pre-defining process.

2. Their responses were vague and unhelpful far from the intelligent.

3. I talked with a few pharmacy chatbots, but their answers were vague and unhelpful far from the intelligent guidance people need.

> Example:  

![](./screenshots/problem6.png "")
![](./screenshots/problem7.png "")
![](./screenshots/problem8.png "")

------------------------------------------------------------------------------------------


---

## Features
MediPal was designed to close this gap by:

1. **Comprehensive Medicine Knowledge**

   * Uses an **Agentic RAG (Retrieval-Augmented Generation) system** to provide accurate, detailed, and up-to-date medicine information for responses.

2. **Real-Time Knowledge Updates**

   * Continuously refreshes its knowledge base to include the latest drug data and treatment guidelines.

3. **Advanced Reasoning & Action**

   * Strong capabilities in reasoning, action, and perception for more natural conversations.

4. **Conversation Memory**

   * Remembers previous interactions to maintain context and prevent broken or repetitive dialogue.

---

## 🗂 Project Structure
The project is organized into four main parts:

1. **Data ETL Pipeline**

   * Scrapes raw medical data
   * Cleans and validates the content with quality checks
   * Schedules regular updates to keep information current

2. **Agentic RAG**

   * Implements properly text chunking and indexing strategies
   * Uses an intelligent retrieval agent to help fetching the most relevant documents for each query

3. **Q&A Agent**

   * Generates accurate, context-aware answers based on the documents retrieved by the RAG component

4. **Front-End**

   * Provides a chat interface that supports both text and voice conversations for an interactive user experience


```text
project-root/
├─ data-pipeline/                 # Data ETL: scraping, cleaning, quality checks, scheduled updates
│  ├─ README.md
│  ├─ etl.py
│  └─ utils/
│
├─ agentic-rag/                   # Chunking, indexing, and intelligent document retrieval
│  ├─ README.md
│  ├─ rag_agent.py
│  └─ indexing/
│
├─ qa-agent/                      # Q&A agent for generating responses from retrieved documents
│  ├─ README.md
│  ├─ qa_agent.py
│  └─ models/
│
├─ frontend/                      # Chat interface with text and voice support
│  ├─ README.md
│  ├─ app/
│  └─ static/
│
├─ tests/                         # Unit and integration tests
│  └─ ...
│
├─ data/                          # Datasets and processed outputs
│  ├─ raw/
│  └─ processed/
│
└─ README.md                       # Main project overview
````

---

## 🚀 Getting Started

### Prerequisites

List all software or tools required.

```bash
# Example
python >= 3.10
node >= 18
```

### Installation

Step-by-step guide:

```bash
git clone https://github.com/yourname/yourproject.git
cd yourproject
pip install -r requirements.txt
```

---

## 🖥 Usage

Provide clear examples:

```bash
python src/main.py --config config.yaml
```

You can also show **inline code** like `python main.py` within a sentence.

---

## ⚙️ Configuration

Explain environment variables, `.env` files, or configuration options.

| Variable      | Description              | Default |
| ------------- | ------------------------ | ------- |
| `API_KEY`     | Key for external service | None    |
| `ENVIRONMENT` | `dev` / `prod`           | `dev`   |

---

## 🔗 Links

* **Documentation:** [Project Docs](https://example.com/docs)
* **Demo:** [Live Demo](https://example.com/demo)

---

## 🧩 Examples

Embed different code languages:

<details>
<summary>Python</summary>

```python
from yourmodule import run
run()
```

</details>

<details>
<summary>JavaScript</summary>

```javascript
import { run } from 'yourmodule';
run();
```

</details>

---

## ✅ Tests

```bash
pytest tests/
```

---

## 🛣 Roadmap

* [ ] Add authentication
* [ ] Implement CI/CD
* [ ] Multi-language support

See the [open issues](https://github.com/yourname/yourproject/issues) for full list.

---

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

* [Awesome Library](https://github.com/some/library)
* [Inspiration Blog Post](https://example.com)

---

### Markdown Tips & Samples

| Element           | Syntax Example                     |
| ----------------- | ---------------------------------- |
| **Bold**          | `**text**`                         |
| *Italic*          | `*text*`                           |
| ~~Strikethrough~~ | `~~text~~`                         |
| Blockquote        | `> quoted text`                    |
| Checklist         | `- [ ] item` or `- [x] done`       |
| Link              | `[title](URL)`                     |
| Image             | `![alt](path_or_URL)`              |
| Footnote[^1]      | `Here is a footnote reference[^1]` |

