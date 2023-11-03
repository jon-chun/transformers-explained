# transformers-explained

## Talk: "Theory, Practice and Future of LLMs" (Transformers Explained)

![alt text]("./one_ai_model_to_rule_transformers_20231103.jpg")

* Jon Chun
* Presented at Denison University
* 2 Nov 2023
* (Slides and Links below)


> If software is eating the world and
> AI is eating software then
> Transformers are eating AI

### <b>Why Focus on Transformer Architecture?</b>

The 2017 <a href="https://arxiv.org/abs/1706.03762">'Attention is all you need'</a> paper by Google was the precedent that led to the November, 30th 2022 launch of ChatGPT. For the first time, anyone could seamlessly and directly interact via natural language with state of the art AI Large Language Models. Over this period of time and continuing to today, we've witnessed rapidly accelerating AI research, roll-out and public awareness from the <a href="https://www.artificial-intelligence-act.com/">EU AI Act</a> to the <a href="https://www.whitehouse.gov/briefing-room/statements-releases/2023/10/30/fact-sheet-president-biden-issues-executive-order-on-safe-secure-and-trustworthy-artificial-intelligence/">Biden Whitehouse Executive Order concerning AI regulation</a>. Concurrently, Large Language Models are also increasing in performance, spawning many capable small open source LLMs and becoming multimodal (Text, Image, Video, Audio, Time Series, etc). A new computational paradigm is forming around these Large Multimodal Models (LMMs) with formalized professional programming ecosystems like <a href="https://github.com/microsoft/semantic-kernel">Microsoft's Semantic Kernel</a> to LMMs embodied in open ended robotic systems like <a href="https://palm-e.github.io/">Google PaLM-e</a>.

Over the past decade I've been focused on AI research and in 2016 co-created the world's first Human-Centered AI curriculum at Kenyon College. Up to 2023, there was a natural progression in our <a href="https://aiforthehumanities.wordpress.com/">'AI for the Humanities'</a> course. Aside from other AI models added in like biologically inspired models, physics-based models, multi-agent game theory simulations, etc., the following order is a rational progression approximating the evolution of DNN architectures. This sequence provides a grounding in the theory and limitations of each generation that inspired innovations in the next.

1. Good old fashion AI (GOFAI) and statistical machine learning models (SVC, XGBoost, etc)
2. Deep Neural Nets (DNNs)
a. Simple fully connected networks (FCN)
b. Variational/auto encoder networks (AE/VAEs)
c. Convolutional neural networks (CNNs) for efficiently learning spatial relations
d. Sequence-to-sequence architectures (RNN, LSTM, GRU, etc) for efficiently learning temporal relations
e. Architectures optimized for generation (GANs and Diffusion Networks)
f. Finally, Transformers with Attention Heads

Today, the growing singular prevelance of the Transformer model is seriously rebalancing priorities. Prior to the rise of Transformers, a variety of specialized architectures arose to optimize for different applications: CNN dominated spatial/vision tasks sequence-to-sequence were used for inherently temporal information like text and audio with occasional hybrids/further specializations. Today, much like the largely relatively undifferentiated repeated structures in the human brain, the Transformer model is proving to be a universalist architecture that excels at most every common task, in every modality and with increasing SOTA Leaderboard performance (e.g. <a href="https://paperswithcode.com/task/image-classification">Image Classification</a> and <a ref="https://paperswithcode.com/task/language-modelling">Language Modeling</a> ).

Although there are numerous theoretical objections to the Transformer architecture in terms of being intrinsically self-limiting and Turing Incomplete (e.g. memory, recursion, symbol manipulation, etc), these issues are being addresses one by one. LLMs are generalizing into Large Multimodal Models (LMMs) or Foundational Models (FM) they are also at the heart of a new abstraction stack around LLMs with additional layers providing solutions to the fundamental limitations of LLMs like hallucination, stochasiticity, innumeracy/symbolic manipulations, stale training data. Efficiencies within Attention Heads and elsewhere constantly expand context windows, superior training/high-quality datasets enhance performance and distill model sizes, paradigms from OS/DB virtual memory paging provide virtually infinite memory, etc. Advanced prompting strategies, fine-tuning and exteral tool use enable sophisticated symbol mainpuation from Python code interpreters, WolframAlpha physics/engineering mathematics to formal math proof systems.

A new AI Computing Paradigm is evolving around an Large Language Models (LLMs). Microsoft, the world's leader in enterprise software development, is focused on creating a development ecosystem from infracture cloud hosting to a complete professional developer ecosystem including: Github repositories, Copilot AI programming assistants, LLMs like GPT4, automated collborative AI Agents in <a href="https://www.microsoft.com/en-us/research/blog/autogen-enabling-next-generation-large-language-model-applications/ ">AutoGen</a> to general .NET-like programming frameworks in <a href="https://learn.microsoft.com/en-us/semantic-kernel/overview/">MS Semantic Kernel</a>. 

The slides and links below are from a talk given for CS faculty and students at Denison University on November 2, 2023 titled "Theory, Practice and Future of LLMs". In our original Kenyon College "AI for the Humanities" course, we would have built a foundation before attempting to explain Tranformers. To minimize issues regarding the diverse backgrounds of our audience and economize on time, the talk focuses on building up a foundation of universal principles before going into details of both the architecture and operation of Transformer models (esp with regard to details like multiple self-attention and cross-attention heads)

<a href="./Theory_Practice_and_Future_of_LLMs_20231102.pdf">Slide Deck in PDF Format</a>

* Slide 7: <a href="https://platform.openai.com/playground">Open AI Playground (Subscription)</a>
* Slide 11: <a href="https://seeing-theory.brown.edu/bayesian-inference/index.html#section1">Seeing Theory: Probability Distributions & Bayesian Inference</a>
* Slide 12: <a href="https://distill.pub/2020/growing-ca/">Growing Neural Cellular Automata</a>
* Slide 16: <a href="https://projector.tensorflow.org/">TensorFlow Embedding Projector</a>
* Slide 27: <a href="https://distill.pub/2017/aia/">Using Artificial Intelligence to Augment Human Intelligence</a>
* Slide 34: <a href="https://playground.tensorflow.org/">Tensorflow Playground</a>
* Slide 36: <a href="https://adamharley.com/nn_vis/cnn/3d.html">CNN Visualization</a>
* Slide 54: <a href="https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ?usp=sharing">BERTViz Tutorial</a>
* Slide 56: <a href="https://www.youtube.com/watch?v=lmepFoddjgQ&t=180s">Visualize what is going on inside Multi-Head Attention Networks (Transformers)</a>
* Slide 68: <a href="https://pi.ai/home">Hey pi.ai Chatbot</a>
