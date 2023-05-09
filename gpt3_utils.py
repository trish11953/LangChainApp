import openai


def get_gpt3_suggestions(api_key, query, num_suggestions=3):
    openai.api_key = api_key
    # Creates a prompt string to ask GPT-3 to generate related questions
    prompt = f'Generate {num_suggestions} related questions to "{query}":'
    for i in range(1, num_suggestions + 1):
        prompt += f"\n{i}."
    # Calls the GPT-3 API with the created prompt
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # Extracts the suggestions from the API response
    suggestions = []
    raw_suggestions = response.choices[0].text.strip().split("\n")[1:]
    for suggestion in raw_suggestions:
        # Cleans up the suggestion text by removing the suggestion number and extra spaces
        suggestions.append(suggestion.strip().lstrip("123."))

    return suggestions
