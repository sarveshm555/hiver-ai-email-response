\# AI Email Suggested-Response System



An AI-powered email suggested-response system built for the Hiver SDE internship coding challenge.



Given an incoming email, the system retrieves similar historical email/reply examples and uses an LLM to generate a concise suggested response.



\## Problem



The goal is to generate useful email replies while learning from a dataset of historical email/reply pairs.



The system should:



1\. Retrieve relevant historical examples.

2\. Use those examples as context for an LLM.

3\. Generate a natural suggested response.

4\. Evaluate the generated response using multiple quality dimensions.



\---



\## Architecture



```text

Incoming Email

&#x20;     |

&#x20;     v

Sentence Transformer

&#x20;     |

&#x20;     v

FAISS Similarity Search

&#x20;     |

&#x20;     v

Top-3 Historical Email/Reply Examples

&#x20;     |

&#x20;     v

LLM Prompt + Retrieved Examples

&#x20;     |

&#x20;     v

Generated Suggested Reply

&#x20;     |

&#x20;     v

Evaluation

