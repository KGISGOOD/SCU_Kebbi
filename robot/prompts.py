from __future__ import annotations
from langchain.prompts import ChatPromptTemplate, PromptTemplate


class PromptFactory:
    @staticmethod
    def context_prompt() -> ChatPromptTemplate:
        template = """{context}

{question}
"""
        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def document_prompt() -> PromptTemplate:
        return PromptTemplate(
            input_variables=["page_content", "episode_name", "Podcast_name"],
            template="內容: {page_content}\n來源: {episode_name}, {Podcast_name}",
        )