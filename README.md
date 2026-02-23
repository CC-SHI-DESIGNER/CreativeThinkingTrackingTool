# CreativeThinkingTrackingTool
# 智能穿戴创意思维追踪工具

This project introduces a novel Intelligent Wearable Creative Thinking Tracking Tool, designed to quantitatively and qualitatively assess the creative thinking processes in early-stage student smart wearable designs. It achieves this by synergistically combining a custom-developed Machine Learning (ML) model with Microsoft's advanced Everything-of-Thoughts (XoT) framework. The core innovation lies in applying a hybrid AI approach to a complex, open-ended domain like design creativity, moving beyond traditional game-theoretic problems.

**Key Scientific Contributions & Innovations:**

*   **Hybrid AI for Creative Assessment:** First-of-its-kind integration of an ML-driven creative assessment model with a Monte Carlo Tree Search (MCTS) guided Large Language Model (LLM) framework (XoT) for iterative design thinking.
*   **Multi-Modal Design State Representation:** Development of a conceptual `canonicalBoard` that unifies diverse design data (textual descriptions, visual elements, structured parameters) into a machine-interpretable feature vector, enabling comprehensive analysis.
*   **Quantification of Design Creativity:** A structured approach to quantify six core dimensions of creativity (Novelty, Utility, Aesthetics, Technical Feasibility, Divergent Thinking, Integration) crucial for smart wearable design.
*   **Iterative Design Guidance:** Leverages XoT's MCTS to explore design modification `actions` and LLM's generative capabilities to provide structured, actionable feedback and revision suggestions, fostering an iterative design cycle.
*   **Explainable AI for Design:** The framework provides not only quantitative scores but also qualitative insights (strengths, weaknesses, recommendations) and can conceptually trace the MCTS 'thought path`, enhancing the transparency and interpretability of creative assessment.

## 2. Cross-Disciplinary Relevance

*   **Design Studies/Human-Computer Interaction (HCI):** Offers a computational tool for analyzing and fostering creativity in design processes. It provides empirical methods to study design ideation, iteration, and the impact of AI feedback on design outcomes. The human-in-the-loop aspect of designers interacting with AI suggestions is central to HCI research.
*   **Educational Technology/Pedagogy:** Provides educators with an objective, data-driven system to track students' creative development in engineering and design curricula. It facilitates personalized feedback, allowing instructors to identify specific areas for improvement in student projects and measure the effectiveness of creative teaching interventions.
*   **Cognitive Science of Creativity:** The XoT framework's ability to model 'thought processes' (MCTS explorations and LLM reasoning) offers a computational analogy for human creative cognition. This tool can be used to generate hypotheses and test theories about how design problems are solved creatively and how individuals respond to constructive criticism.
*   **Innovation Studies/Management:** Provides insights into the dynamics of innovation, particularly in product development. By tracking novelty and integration, it can shed light on factors contributing to successful innovation in technology-driven domains like smart wearables.

## 3. Academic Value and Methodology

**Academic Value:**

*   **Bridging AI and Creativity:** Addresses the long-standing challenge of computationally assessing and enhancing human creativity, offering a robust framework for future research in AI-augmented design.
*   **Empirical Data Generation:** Lays the groundwork for generating empirical data on design iteration patterns, the impact of specific design 'actions', and the evolution of creative attributes over time.
*   **Tool for Research & Practice:** Serves as a foundational tool for researchers studying creativity, design pedagogy, and human-AI collaboration, while also offering practical benefits for design education and R&D.

**Methodology - Hybrid AI Approach:**

Our methodology integrates several advanced AI paradigms:

1.  **Multi-Modal Feature Engineering:** Raw design inputs (text, images, structured data) are processed by specialized encoders (NLP models for text, CNNs for images, etc.) to form a unified `canonicalBoard` feature vector. This `canonicalBoard` serves as the abstract state representation for the XoT framework.
2.  **Reinforcement Learning (RL) with MCTS (XoT Core):** The XoT framework orchestrates a Monte Carlo Tree Search (MCTS) to explore the vast design space. MCTS iteratively selects `creative actions` (e.g., "change material", "add sensor") that modify the `canonicalBoard`, effectively simulating design evolution.
3.  **Neural Network (NN) Policy/Value Prediction:** A custom Multi-Head Neural Network (`CreativeDesignNNet`) acts as the `NeuralNet` component within MCTS. It predicts a policy ($\pi$) over possible `creative actions` and a value ($v$) representing the estimated creative potential of a given design state (`canonicalBoard`). This guides MCTS towards promising design paths.
4.  **Large Language Model (LLM) for Reasoning and Feedback:** An LLM is integrated to contextualize MCTS 'thoughts' (action sequences), generate natural language design descriptions, provide comprehensive key insight reports (strengths, weaknesses), and suggest specific, human-understandable recommendations for design revisions.
5.  **Feedback Loops for Refinement:** The LLM-generated solutions are evaluated by a simulated ML-based parser that provides quantitative creative dimension scores. This feedback closes the loop, allowing the XoT framework to guide the LLM to revise designs iteratively until desired creative criteria are met.

## 4. Guide for Running and Verifying the Tool

To run and verify the Intelligent Wearable Creative Thinking Tracking Tool prototype, please follow these steps:

**1. Clone the Repository:**

Navigate to your desired directory and clone the project repository:

```bash
git clone https://github.com/microsoft/Everything-of-Thoughts-XoT.git
cd Everything-of-Thoughts-XoT

## 项目概述

本项目旨在开发一个智能工具，通过结合机器学习模型与Microsoft的Everything-of-Thoughts (XoT) 框架，对学生早期智能穿戴设计中的创意思维进行多维度追踪、评估和迭代优化。该工具能够量化评估设计的"新颖性"、"实用性"、"美观度"、"技术可行性"、"发散性思维"和"整合性"等关键维度，并提供结构化的反馈和具体的改进建议，从而辅助学生培养创新能力并优化设计流程。

## 特性

*   **多维度创意评估**：提供6个核心创意维度的量化评分（0-100分）。
*   **整体创意评级**：将设计分类为"高创意"、"中创意"或"低创意"。
*   **结构化洞察报告**：生成包含设计亮点、改进空间和具体建议的文本报告。
*   **MCTS-LLM驱动的迭代**：利用XoT框架，通过蒙特卡洛树搜索（MCTS）探索设计空间，并结合大型语言模型（LLM）进行"思想"生成和设计方案的修订。
*   **多模态数据处理（模拟）**：原型展示了将文本描述、视觉资料和结构化数据编码为统一特征向量的能力。
*   **模块化与可扩展性**：清晰定义的接口，便于未来集成更先进的ML模型和LLM。

## 安装指南

1.  **克隆Everything-of-Thoughts-XoT仓库**：

    ```bash
    git clone https://github.com/microsoft/Everything-of-Thoughts-XoT.git
    cd Everything-of-Thoughts-XoT
    ```

2.  **设置Python环境**：

    建议使用`conda`或`venv`创建虚拟环境。

    ```bash
    conda create -n creativity_env python=3.9
    conda activate creativity_env
    # 或者
    python -m venv creativity_env
    source creativity_env/bin/activate
    ```

3.  **安装依赖**：

    ```bash
    pip install -r requirements.txt
    ```

    注意：`requirements.txt` 文件包含了运行本项目所需的所有Python库。其中`torch`可能需要根据您的CUDA版本进行特定安装。

4.  **设置OpenAI API Key (如果使用真实LLM)**：

    如果您计划使用真实的OpenAI LLM，请确保您的OpenAI API Key已设置为环境变量 `OPENAI_API_KEY`。

    ```bash
    export OPENAI_API_KEY="YOUR_API_KEY"
    ```

## 使用方法

本项目原型通过运行 `IntegratedXoTPrototype.py` 脚本来演示其功能。该脚本模拟了从初始设计评估到MCTS驱动的迭代和LLM反馈的整个工作流程。

1.  **运行原型演示**：

    ```bash
    python IntegratedXoTPrototype.py
    ```

2.  **查看输出**：

    脚本将在控制台输出模拟的设计评估过程，包括LLM的初始洞察、MCTS生成的"思想"、ML模型评估的创意分数、整体评级以及修订建议。

3.  **自定义模拟数据**：

    您可以修改 `IntegratedXoTPrototype.py` 文件中的 `simulated_design_data` 字典，以模拟不同的智能穿戴设计输入。

## 核心组件

本项目集成的核心组件及其作用如下：

*   **`CreativeDesignGame.py`**：
    *   实现了XoT框架的`Game`接口，将智能穿戴设计抽象为"游戏"状态（`canonicalBoard`）。
    *   定义了"创意动作空间"，模拟设计修改对`canonicalBoard`的影响（`getNextState`）。
    *   包含一个模拟的多模态数据预处理器 `preprocess_design_data`，用于将原始设计数据转化为统一的特征向量。

*   **`CreativeDesignNNet.py`**：
    *   实现了XoT框架的`NeuralNet`接口，充当我们自定义的机器学习模型。
    *   内部是一个多头神经网络，接收`canonicalBoard`并输出"创意动作策略" (`pi`) 和"创意价值" (`v`)，指导MCTS的探索。

*   **`CreativeDesignPrompter.py`**：
    *   负责将设计状态、MCTS"思想"和ML模型反馈转化为自然语言提示（prompts），供LLM使用，以生成评估报告和修订建议。

*   **`CreativeDesignParser.py`**：
    *   负责解析LLM的输出，并通过模拟ML评估模块，生成创意维度分数、判断设计是否"合格"，并提取结构化的反馈信息。
    *   将MCTS的数值动作转化为人类可读的"思想"文本。

*   **`IntegratedXoTPrototype.py`**：
    *   一个端到端的模拟脚本，整合了上述所有自定义组件和XoT框架的核心逻辑。
    *   通过 `SimulatedXoTSolver` 类，演示了从设计提交、ML评估、MCTS驱动的"思想"生成到LLM迭代修订的完整流程。

## 项目贡献

本项目通过以下方式对智能穿戴设计领域的创意追踪做出贡献：

*   **多模态创意评估框架**：提出了一个整合多模态数据（文本、视觉、结构化）进行创意思维评估的框架。
*   **结合MCTS与LLM**：创新性地将XoT（一个结合MCTS和LLM的框架）应用于开放域的创意设计迭代过程，而非传统的封闭"游戏"。
*   **量化与定性反馈**：实现了量化分数（创意维度）与定性报告（LLM生成）相结合的评估机制，为设计师提供全面且可操作的反馈。
*   **设计迭代支持**：通过MCTS驱动的"思想"和LLM修订建议，为学生设计师提供了一个支持持续迭代和优化的智能辅助工具。

## 未来工作

*   **真实的特征编码器**：开发功能完备的多模态特征编码器，集成先进的NLP（如BERT）和CV（如CNN）模型，处理真实的设计数据。
*   **`CreativeDesignGame`和`CreativeDesignNNet`的完善**：细化游戏规则，实现更真实的"创意动作"对设计状态的影响，并通过大量MCTS自博弈数据训练更精确的`CreativeDesignNNet`。
*   **LLM深度集成**：将真实的OpenAI LLM集成到XoT框架中，优化提示工程，生成更富有洞察力和上下文感知的报告和建议。
*   **用户界面开发**：基于已设计的UI/UX规划，构建一个功能完整、用户友好的前端界面，实现与后端模型的实时交互。
