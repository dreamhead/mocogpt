# MocoGPT

MocoGPT is an GPT mock server, which can be used to test your GPT API calls.

* Prompt and Response Generation: Setup response generation based on prompt as you wish.
* Steaming Support: Support streaming response generation.
* OpenAI API Support: Work with OpenAI API.
* Standalone Server: Run MocoGPT as a standalone server.

## Usage

MocoGPT can be used as a library, or a standalone server.

### As a Library

You can use MocoGPT as a library as following:

```python
from openai import OpenAI
from mocogpt import gpt_server, prompt, content

def test_reply():
    server = gpt_server(12306)
    server.request(prompt("Hi")).response(content("How can I assist you?"))

    with server:
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert response.choices[0].message.content == "How can I assist you?"
```

In this case, we setup `gpt_server` to listen on port `12306`. When `prompt` is "Hi", response `content` will be "How can I assist you?".

We use OpenAI library to send a chat request to our mock GPT server. 
Here we should set base_url to http://localhost:12306/v1 and verify the response.

### As a standalone server

MocoGPT can be run as a standalone server.

You can write your own configuration:
```json
[
    {
        "prompt": "Hi",
        "response": "How can I assist you?"
    }
]
```
(config.json)
    
Then run MocoGPT server:

```bash
$ pip install mocogpt[cli]
$ mocogpt start config.json --port 12306
```

Now, you can send a chat request to http://localhost:12306/v1.


