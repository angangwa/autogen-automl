"""
TODO:
- https://microsoft.github.io/autogen/stable/_modules/autogen_core/model_context/_buffered_chat_completion_context.html#BufferedChatCompletionContext
- Extent BufferedChatCompletionContext to support unbounded context for the model itself, but limited context from the other agents.
- https://microsoft.github.io/autogen/stable/_modules/autogen_core/model_context/_unbounded_chat_completion_context.html#UnboundedChatCompletionContext
- Need to process LLMMessage: https://microsoft.github.io/autogen/stable/_modules/autogen_core/models/_types.html
    - e.g. Remove all tool call context from the other agents
- Problm - funtion call messages not tracking source, how do I remove them based on the source? Create an issue?
"""