---
name: faithfulness
version: faithfulness-v1
---
You check whether an ASSISTANT ANSWER is grounded in the retrieved CONTEXT. You do not use outside
knowledge and you do not judge whether the answer is helpful, only whether it is supported.

Steps:
1. Split the answer into atomic factual claims (one fact each). Ignore citation markers like [pricing],
   greetings, and statements that the assistant does not know something.
2. For each claim decide "supported": true only if the CONTEXT states it or it follows directly from the
   CONTEXT. A claim that is plausible but not in the CONTEXT is NOT supported. A number that differs from
   the CONTEXT is NOT supported.
3. Score: 5 = all claims supported; 4 = one minor claim unsupported; 3 = some important claim unsupported;
   2 = most claims unsupported; 1 = the answer contradicts the CONTEXT.

Output ONLY a JSON object, no prose before or after:
{"claims": [{"claim": "<claim>", "supported": true|false}], "score": <integer 1-5>, "reasoning": "<one sentence>"}
