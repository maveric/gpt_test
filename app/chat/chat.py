import openai
import requests
import json
from typing import List, Dict, Union
import uuid
from .plugins.plugin import PluginInterface
from .plugins.webSearch import WebSearchPlugin
from .plugins.pythoninterpreter import PythonInterpreterPlugin
from .plugins.webscraper import WebScraperPlugin

class Conversation:
    """
    This class represents a conversation with the ChatGPT model.
    It stores the conversation history in the form of a list of
    messages.
    """
    def __init__(self):
        self.conversation_history: List[Dict] = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

GPT_MODEL = "gpt-3.5-turbo-16k-0613"

SYSTEM_PROMPT = """
    You are a helpful AI assistant. You answer the user's queries.
    When you are not sure of an answer, you take the help of
    functions provided to you. 
    If you get a response of 'Not result written to stdout. Please print result on stdout'
    it means that you did not print anything to the console. You must print to console to
    get a response when using the python interpreter.
    You may also use the websearch and webscraper functions to get information from the web. They're best used 
    used together to find information on a website and then scrape it.
    NEVER make up an answer if you don't know, just respond
    with "I don't know" when you don't know.
"""

class ChatSession:
    """
    Represents a chat session.
    Each session has a unique id to associate it with the user.
    It holds the conversation history
    and provides functionality to get new response from ChatGPT
    for user query.
    """
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation = Conversation()
        self.plugins: Dict[str, PluginInterface] = {}
        self.register_plugin(WebSearchPlugin())
        self.register_plugin(PythonInterpreterPlugin())
        self.register_plugin(WebScraperPlugin())
        self.conversation.add_message("system", SYSTEM_PROMPT)


    def register_plugin(self, plugin: PluginInterface):
        """
        Register a plugin for use in this session
        """
        self.plugins[plugin.get_name()] = plugin

    def _get_functions(self) -> List[Dict]:
        """
        Generate the list of functions that can be passed to the chatgpt
        API call.
        """
        return [self._plugin_to_function(p) for
                p in self.plugins.values()]

    def _plugin_to_function(self, plugin: PluginInterface) -> Dict:
        """
        Convert a plugin to the function call specification as
        required by the ChatGPT API:
        https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
        """
        function = {}
        function["name"] = plugin.get_name()
        function["description"] = plugin.get_description()
        function["parameters"] = plugin.get_parameters()
        return function
    
    def _execute_plugin(self, func_call) -> str:
        """
        If a plugin exists for the given function call, execute it.
        """
        # print(f"Executing plugin for {func_call}")
        func_name = func_call.get("name")
        print(f"Function name: {func_name}")
        if func_name in self.plugins:
            arguments = json.loads(func_call.get("arguments"))
            plugin_response = self.plugins[func_name].execute(**arguments)
        else:
            plugin_response = {"error": f"No plugin found with name {func_call}"}

        # We need to pass the plugin response back to ChatGPT
        # so that it can process it. In order to do this we
        # need to append the plugin response into the conversation
        # history. However, this is just temporary so we make a
        # copy of the messages and then append to that copy.
        messages = list(self.conversation.conversation_history)
        messages.append({"role": "function",
                         "content": json.dumps(plugin_response),
                         "name": func_name})
        # print(f"Plugin response - {func_name}:{plugin_response}")
        next_chatgpt_response = self._chat_completion_request(messages)
        # print(f"Next ChatGPT response: {next_chatgpt_response}")

        # If ChatGPT is asking for another function call, then
        # we need to call _execute_plugin again. We will keep
        # doing this until ChatGPT keeps returning function_call
        # in its response. Although it might be a good idea to
        # cut it off at some point to avoid an infinite loop where
        # it gets stuck in a plugin loop.
        if isinstance(next_chatgpt_response, dict):
            if next_chatgpt_response.get("function_call"):
                return self._execute_plugin(next_chatgpt_response.get("function_call"))
            return str(next_chatgpt_response.get("content"))
        # else:
        #     try:
        #         print(f'Attempting to print choices: {next_chatgpt_response.text}')
        #     except Exception as e:
        #         raise Exception(f"ChatGPT response is not a dict, returned {next_chatgpt_response} with error: {e}")
            
        return "Error: Something went wrong here"
    
    def get_messages(self) -> List[Dict]:
        """
        Return the list of messages from the current conversation
        """
        # Exclude the SYSTEM_PROMPT when returning the history
        if len(self.conversation.conversation_history) == 1:
            return []
        return self.conversation.conversation_history[1:]
    
    def get_chatgpt_response(self, user_message: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        self.conversation.add_message("user", user_message)
        try:
            chatgpt_response = self._chat_completion_request(
                self.conversation.conversation_history
                )
            if isinstance(chatgpt_response, dict):

                if chatgpt_response.get("function_call"):
                    print(f'function call: {chatgpt_response.get("function_call")}')
                    chatgpt_message = self._execute_plugin(
                        chatgpt_response.get("function_call"))
                else:
                    # print(f'content: {chatgpt_response.get("content")}')
                    chatgpt_message = str(chatgpt_response.get("content"))
            else:
                raise Exception(f"ChatGPT response is not a dict, returned {chatgpt_response}")
            self.conversation.add_message("assistant", chatgpt_message)
            return chatgpt_message
        except Exception as e:
            print(e)
            return "something went wrong"

    def _chat_completion_request(self, messages: List[Dict]) -> Union[Dict, Exception]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + (openai.api_key or ""),
        }
        json_data = {"model": GPT_MODEL,
                     "messages": messages,
                     "temperature": 0.7}
        if len(self.plugins) > 0:
            json_data["functions"] = self._get_functions()

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            # print(f"ChatGPT response: {response.json()}")
            if("error" in response.json()):
                print(f'Error: {response.json()["error"]}')
                return response.json()["error"]['message']
            return response.json()["choices"][0]["message"]
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e