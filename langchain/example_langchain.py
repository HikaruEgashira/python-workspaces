# %% Simple
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
message = llm.invoke("how can langsmith help with testing?")
print(message)

# %% Prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are world class technical documentation writer."), ("user", "{input}")]
)
chain = prompt | llm
message = chain.invoke({"input": "how can langsmith help with testing?"})
print(message)

# %% Output Parser
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
message = chain.invoke({"input": "how can langsmith help with testing?"})
print(message)
