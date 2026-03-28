Here's the full content of the PDF:

---

**NATURAL LANGUAGE PROCESSING PROJECT BRIEF**
Project context document

---

**Multi-Agent QA on Financial Documents**
*Challenge Theme: From Raw Financial Reports to Structured, Reasoned Answers*

**Your Task:**
Financial documents—like annual reports, earnings statements, and regulatory filings—are long, complex, and multimodal. They include text, tables, and figures, all of which encode critical financial insights. Your task is to design a hierarchical multi-agent system that can:
1. Parse and structure financial documents.
2. Retrieve relevant information in response to complex user queries.
3. Reason across multiple modalities and subtasks.
4. Produce transparent reasoning logs showing which agents and tools were used.

The goal is not just to answer queries correctly but to demonstrate explainable reasoning through a modular, coordinated multi-agent architecture.

**Why This Matters:**
- Financial analysts, auditors, and regulators often spend weeks digging through 100+ page reports.
- Machines struggle with financial documents because they don't just contain plain text — they have tables, figures, and multimodal content that encode critical insights.

By building structured representations, you enable advanced applications like automated Q&A, compliance checks, and trend analysis.

---

**Task 1: Financial Document Chunking and Storage**

You are given financial documents, which are typically long and multimodal (containing text, tables, and figures). Your task is to:
- Chunk the documents in a way that preserves important structure and context.
- Store the chunks effectively so that they are optimized for retrieval in downstream query answering (Task 2).

Decide how to handle multimodal content within the chunks to maximize reasoning and retrieval efficiency.

---

**Task 2: Dynamic Multi-Agent Query Answering on Financial Documents**

Using the chunked and stored financial documents from Task 1, your goal is to design a dynamic multi-agent system capable of handling unpredictable, user-driven queries, based on the **Swarm architecture**.

**Architecture Overview**

- **Dynamic Multi-Agent Swarm Workflow**
  - There will be multiple specialized agents, each with access to an array of tools.
  - These agents will communicate with each other through hand-offs, with each agent selecting which other agent to hand over the control after its turn ends.
  - The interaction history and hand-offs should be stored and everyone should be provided access to the same.
  - All agents should have access to context/previous agent actions, along with proper chat-format templates of prompts defining agents, and detailed prompts for each agent along with tool descriptions, and handling of tool calling and execution.

- **Memory Management**
  - History must be maintained of the conversation of the human user with the agentic system of previous queries, to enable rapid answering of queries similar to the ones asked earlier.

**Agent Roles & Example Tools**

| Agent | Example Tools | Role |
|---|---|---|
| Information Agent | Vector Index of parsed doc data, web search | Fetches relevant document chunks |
| Web Search Agent | Web Search API | Retrieves up-to-date relevant information |
| Math & Analysis Agent | Internal computation / statistical reasoning | Performs essentially every math operation that can be programmatically expressed |
| Summarization Agent | LLM summarization | Condenses text-based content |

*Note: These are example agents and tools; participants can design additional agents and tools to suit the query requirements.*

---

**Query Example:**

**User Query:**
"Compare the YoY revenue growth and R&D spending between 2021 and 2022, and summarize the risks affecting future revenue."

**Control Flow:**
1. Retriever Agent → Fetch revenue and R&D tables for 2021 and 2022.
2. Table Agent → Extract numeric values and calculate necessary ratios.
3. Math and Reasoning Agent → Compute YoY growth percentages for revenue and R&D.
4. Retriever Agent → Fetch risk-related sections from the document.
5. Aggregator Agent → Aggregate numeric and textual results into a structured final answer.

Shared Memory → Update with all intermediate results and task completion status for potential reuse.

**Deliverables:**
- **Final Answer:** Concise and human-readable response to the user query.
- **Structured Log:** JSON capturing:
  - Subtask decomposition
  - Agent assignments
  - Tools invoked (with input/output)

**Example Log Schema:**
```json
{
  "query": "<user query>",
  "trace": [
    { "agent": "Retriever", "tool": "...", "input": "...", "output": "...", "handoff-to": "..." },
    { "agent": "Table Agent", "tool": "...", "input": "...", "output": "...", "handoff-to": "..." },
    ...
  ],
  "final_answer": "<concise human-readable answer>"
}
```

---

**Datasets:**
- **FinanceBench** and **Financial Q&A - 10k** datasets (for reference, pre-parsed or raw formats). The finance 10-k reports referred to here can be obtained from the internet (look it up on Google).

Participants are free to use these datasets as a reference, or they may use other financial datasets of their choice.

---

**Evaluation Criteria:**
- **Pipeline Explainability:** Clear demonstration of how each sub-agent and tool contributed to solving the query.
- **Memory Management & Caching:** Evaluate how efficiently the system stores and reuses previous context to handle sequential or related queries without repeating the entire workflow.
- **Error Handling & Fallback:** Assess the system's ability to gracefully handle errors, including tool/agent failures or unavailable data, and to fallback to alternative strategies to still produce a coherent answer.
- **Multimodal Reasoning:** Evaluate how effectively the system processes and integrates information from multiple modalities (text, tables, figures/charts) to answer queries accurately and coherently.
- **Complex Query Handling:** Ability to handle reasoning-heavy and textual queries spanning multiple document sections or modalities.
- **Optional Bonus:** Fine-tuning of sub-agents for domain-specific tasks, multimodal reasoning approaches, or creative agent/tool design.

---

**Submission Guidelines**
- Upload all code along with a detailed readme describing your approach to both tasks in a GitHub repository, along with usage instructions.
- Also, provide a report explaining in detail your approach, with necessary diagrams, statistics to support your solution.
- Link to the WhatsApp group: WhatsApp Community
- In case of any doubts, contact any of the two numbers below:
  - Tejbir - 9034705165
  - Bhaagyesh - 7428647019
