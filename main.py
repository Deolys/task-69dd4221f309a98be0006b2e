from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

class TaskCard(BaseModel):
    title: str = Field(..., description="Краткое название задачи")
    subject: str = Field(..., description="Тема или предмет задания")
    deadline_hint: Optional[str] = Field(None, description="Указание срока сдачи в свободной форме")
    deliverable_type: str = Field(..., description="Тип сдаваемого материала (отчёт, код и т.д.)")
    grading_hints: List[str] = Field(default_factory=list, description="Ключевые критерии оценки")

def build_chain(llm):
    parser = PydanticOutputParser(pydantic_object=TaskCard)
    prompt_template = PromptTemplate(
        template="""
Ниже приведено описание задания от преподавателя. Ваша задача — извлечь из него структурированные данные и вернуть их в формате JSON, соответствующем модели TaskCard.

Описание:
{input_text}

Ответ должен содержать только поля: title, subject, deadline_hint, deliverable_type, grading_hints.
Формат ответа:
{format_instructions}
""",
        input_variables=["input_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt_template | llm | parser

def main():
    # Инициализация модели (используется переменная окружения OPENAI_API_KEY)
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    chain = build_chain(llm)

    example_text = (
        "Сдайте к пятнице мини‑отчёт по LangChain: 2 страницы, упор на агентов. "
        "Оценка: за полноту и за пример кода."
    )

    result = chain.invoke({"input_text": example_text})
    print("Валидированный объект:")
    print(result.model_dump())
    print("\nКраткая сводка:")
    print(f"Тема: {result.subject}")
    print(f"Заголовок: {result.title}")
    print(f"Срок сдачи: {result.deadline_hint or 'не указан'}")
    print(f"Тип сдаваемого материала: {result.deliverable_type}")
    print("Критерии оценки:", ", ".join(result.grading_hints))

if __name__ == "__main__":
    main()
