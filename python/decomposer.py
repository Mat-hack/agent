"""Turn ONE natural-language request that describes many things into an
ordered list of single-entity create/update operations.

    "create project ekam with a package ecom, value types Customer (id, name,
     email) and Order (id, total), and a payments service"

becomes an ordered plan:

    1. create Package  — "package ecom"
    2. create ValueType — "Customer value type in package ecom with id, name, email"
    3. create ValueType — "Order value type in package ecom with id, total"
    4. create Service   — "payments service in package ecom"
    5. create Project   — "project ekam selecting package ecom"

The order matters: a parent must exist before the child that references it,
because each step is executed in turn and the next step's grounding lists the
parent that was just created. The decomposer only PLANS; execution is the
caller's job (see nl_agent.run_plan).
"""
from langchain_groq import ChatGroq

import config


def _plan_schema(entities: list[str]) -> dict:
    return {
        'title': 'emit_plan',
        'type': 'object',
        'properties': {
            'steps': {
                'type': 'array',
                'description': 'ordered operations, parent-first.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'operation': {
                            'type': 'string',
                            'enum': ['create', 'update'],
                        },
                        'entity': {
                            'type': 'string',
                            'enum': entities,
                        },
                        'intent': {
                            'type': 'string',
                            'description': (
                                'a complete, standalone instruction for THIS one '
                                'entity, naming its parent by name.'
                            ),
                        },
                    },
                    'required': ['operation', 'entity', 'intent'],
                },
            }
        },
        'required': ['steps'],
    }


_DECOMPOSER_SYSTEM = (
    'You break ONE natural-language request about a low-code backend generator '
    'into an ORDERED list of single-entity operations. Each step targets '
    'exactly one record.\n'
    '\n'
    'RULES:\n'
    '1. One step per distinct record. If the request mentions several value '
    'types, services, or actions, emit one step for EACH — never bundle them.\n'
    '2. Order steps so every parent is created BEFORE anything that references '
    'it. Dependency order:\n'
    '   - a Package before any ValueType / Service that lives in it\n'
    '   - a Service before any Action on it\n'
    '   - the Project LAST, after the packages it selects already exist\n'
    '3. Each step\'s "intent" must be a self-contained instruction for that '
    'one record, naming its parent by name (e.g. "Customer value type in '
    'package ecom with fields id, name, email"). The downstream generator only '
    'sees that one intent, so it must carry every detail for that record.\n'
    '4. Use only the allowed entity enums. Default operation is "create" unless '
    'the request clearly asks to modify an existing record.\n'
    '5. A project, package, or service named only as the LOCATION/parent of '
    'what is being created (e.g. "in project work1", "in package ecom", "on '
    'the payments service") is assumed to ALREADY EXIST. Do NOT emit a step to '
    'create it. Only create an entity when the user explicitly asks to create / '
    'make / add THAT entity itself.\n'
    'Plan only — do not invent records the request did not ask for.'
)


def decompose(nl: str, entities: list[str]) -> list[dict]:
    """Return an ordered list of {operation, entity, intent} steps."""
    llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0)
    structured = llm.with_structured_output(
        _plan_schema(entities), method='function_calling'
    )
    result = structured.invoke(
        [('system', _DECOMPOSER_SYSTEM), ('human', f'Request: {nl}')]
    )
    return result.get('steps', [])
