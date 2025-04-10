<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/spritz-group/LLM-CVSS">
    <img src="assets/logo.png" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">Can LLMs Classify CVEs?</h1>

  <p align="center">
    Investigating LLMs Capabilities in Computing CVSS Vectors
    <br />
    <a href="https://github.com/spritz-group/LLM-CVSS"><strong>Paper in progress »</strong></a>
    <br />
    <br />
    <a href="https://www.math.unipd.it/~fmarchio/">Francesco Marchiori</a>
    ·
    <a href="https://www.math.unipd.it/~donadel">Denis Donadel</a>
    ·
    <a href="https://www.math.unipd.it/~conti/">Mauro Conti</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
  </ol>
</details>

<div id="abstract"></div>

## 🧩 Abstract

>Common Vulnerability and Exposure (CVE) records are essential in cybersecurity, providing unique identifiers for publicly known vulnerabilities in software and systems. CVEs are critical for managing and prioritizing security risks, with each vulnerability being assigned a Common Vulnerability Scoring System (CVSS) score to aid in their assessment and remediation. However, variations in CVSS scores between different stakeholders often occur due to subjective interpretations of certain metrics. Moreover, the large volume of new CVEs published daily highlights the need for automation in this process to generate accurate and consistent scores. While previous studies explored various approaches to automation, the role of Large Language Models (LLMs), which have gained significant attention in recent years, remains largely unexplored. In this paper, we investigate the potential of LLMs for CVSS evaluation, focusing on their ability to generate accurate CVSS scores for newly reported vulnerabilities. We explore different prompt engineering strategies to optimize the performance of LLMs and compare their results with embedding-based models, where embeddings are generated and then classified using supervised learning approaches. Our findings suggest that while LLMs show promise in certain aspects of CVSS evaluation, traditional embedding-based systems surprisingly perform better when assessing more subjective components, such as the evaluation of confidentiality, integrity, and availability impacts. These results underline the complexity of vulnerability scoring and emphasize the need for continued exploration of hybrid approaches that combine the strengths of both methods.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

To replicate our result or start using the LLMs and embedding models, start by cloning the repository.

```bash
git clone https://github.com/spritz-group/LLM-CVSS.git
cd LLM-CVSS
```

Then, install the required Python packages by running the following command. We recommend setting up a dedicated environment to run the experiments.

```bash
pip install -r requirements.txt
```

The `llms.py` script also includes OpenAI models. You can choose to not use them by toggling `useOpenAI` to false. If instead you want to use GPT models, you should add your own API key with the following command.

```bash
export OPENAI_API_KEY=<your openai key>
```

There is also the possibility to run models from OpenRouter. Since they can use the same OpenAI client, you should manually overwrite the API key in the `llms.py` script.

<p align="right"><a href="#top">(back to top)</a></p>