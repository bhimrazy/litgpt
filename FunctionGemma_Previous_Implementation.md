# FunctionGemma Prompt Style - Previous Implementation

This document contains the previous manual implementation of the FunctionGemma prompt style before switching to the Hugging Face chat template.

## Previous Implementation (Manual Formatting)

```python
class FunctionGemma(PromptStyle):
    def apply(self, prompt: str, *, sys_prompt: Optional[str] = None, **kwargs: str) -> str:
        # FunctionGemma is designed for function calling (tool use)
        # For proper function calling, use the chat_template.jinja from Hugging Face
        # This basic implementation is for simple text generation only
        if sys_prompt:
            # Include system prompt as developer message for function calling setup
            return f"<start_of_turn>developer\n{sys_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        # FunctionGemma has additional stop tokens for function calls
        stop_tokens = [tokenizer.eos_id]
        try:
            # Add function response end token as a stop token
            stop_tokens.append(tokenizer.token_to_id("<end_function_response>"))
        except ValueError:
            pass  # Token might not exist
        return (stop_tokens,)
```
```python
class FunctionGemma(PromptStyle):
    def apply(self, prompt: str, *, sys_prompt: Optional[str] = None, **kwargs: str) -> str:
        # FunctionGemma is designed for function calling, not general chat
        # For proper function calling, use the chat_template.jinja from Hugging Face
        # This basic implementation is for simple text generation only
        if sys_prompt:
            # Include system prompt as developer message for function calling setup
            return f"<start_of_turn>developer\n{sys_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        # FunctionGemma has additional stop tokens for function calls and responses
        stop_tokens = [tokenizer.eos_id]
        try:
            # Add function-related tokens as stop tokens
            stop_tokens.extend([
                tokenizer.token_to_id("<end_function_call>"),
                tokenizer.token_to_id("<end_function_response>"),
            ])
        except (ValueError, KeyError):
            pass  # Tokens might not exist
        return (stop_tokens,)
```
## Issues with Previous Implementation

1. **Limited Function Calling Support**: Only basic chat formatting, no actual tool/function calling workflow
2. **Manual Formatting**: Custom string formatting instead of using the official chat template
3. **Missing Special Tokens**: Didn't handle `<start_function_declaration>`, `<start_function_call>`, etc.
4. **No Tool Integration**: Couldn't properly format tool definitions or handle function calling sequences

## What Was Missing

The previous implementation was essentially just a basic chat format like regular Gemma, but FunctionGemma requires:

- **Tool Definitions**: Using `<start_function_declaration>` and `<end_function_declaration>` tokens
- **Function Calls**: Using `<start_function_call>` and `<end_function_call>` tokens
- **Function Responses**: Using `<start_function_response>` and `<end_function_response>` tokens
- **String Escaping**: Using `<escape>` tokens for string values in structured data
- **Complete Workflow**: Support for the full function calling lifecycle

## Current Implementation (Chat Template)

The current implementation uses the official Hugging Face chat template (`apply_chat_template()`) which handles all the complex formatting automatically and provides full function calling support.

### Key Improvements

1. **Official Chat Template**: Uses the source of truth from Hugging Face
2. **Full Function Calling**: Supports complete tool definition, call, and response workflow
3. **Proper Token Handling**: Correctly uses all special tokens (`<start_function_*>`, `<end_function_*>`, `<escape>`)
4. **Automatic Formatting**: No manual string manipulation needed
5. **Perfect Compatibility**: Output matches Hugging Face transformers exactly

### Usage

```python
# Basic chat
prompt_style.apply("Hello, how are you?")

# Function calling with tools
tools = [{"type": "function", "function": {...}}]
prompt_style.apply("What's the weather?", tools=tools, tokenizer=tokenizer)
```

## Why the Change Was Necessary

FunctionGemma is specifically designed for function calling, not general chat. The previous implementation only provided basic chat formatting, which didn't utilize FunctionGemma's main purpose. By switching to the official chat template, we now have:

- ✅ Full function calling workflow support
- ✅ Proper tool definition formatting
- ✅ Correct special token usage
- ✅ Compatibility with Hugging Face ecosystem
- ✅ Future-proof implementation

## Reference

- [FunctionGemma Documentation](https://ai.google.dev/gemma/docs/functiongemma)
- [Hugging Face Chat Template](https://huggingface.co/google/functiongemma-270m-it)
- [Function Calling Best Practices](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)</content>
<parameter name="filePath">/Users/bhimrajyadav/Developer/LightningAI/litgpt/FunctionGemma_Previous_Implementation.md
