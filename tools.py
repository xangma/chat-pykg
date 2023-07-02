# chat-pykg/tools.py
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar
from langchain.agents import Tool
from langchain.utilities import BashProcess, GoogleSearchAPIWrapper, PythonREPL

def get_tools(g_api_key) -> List[Tool]:
    # Tools
    tools = []

    # Search
    if g_api_key:
        search = GoogleSearchAPIWrapper(search_engine="google")
        tools.append(
            Tool(
            name = 'Google Search',
            func = search.run,
            description="useful for when you need to search the web. Input should be a fully formed question."
            )
        )
    # Bash
    bash = BashProcess()
    tools.append(
        Tool(
        name = 'Bash',
        func = bash.run,
        description="useful for when you need to run bash commands. Input should be a fully formed bash command."
        )
    )
    # Python REPL
    python_repl = PythonREPL()
    tools.append(
        Tool(
        name = 'Python REPL',
        func = python_repl.run,
        description="useful for when you need to run python commands. Input should be a fully formed python command."
        )
    )
    return tools