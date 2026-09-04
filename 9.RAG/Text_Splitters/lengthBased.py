from langchain_text_splitters import CharacterTextSplitter

text = """
Apple is preparing for its highly anticipated iPhone 18 Pro launch on September 9, with leaks indicating significant upgrades like a 2nm A20 Pro chip, a larger battery, and a new C2 modem.

Corporate AI shifts and national policies are also making waves today. OpenAI recently ended its partnership with the AI coding assistant Cursor while restructuring its leadership by bringing on former Meta executive Sandhya Devanathan.

On a national level, Thailand introduced an innovative "AI passport" program designed to provide its citizens with free access to premium artificial intelligence tools, including ChatGPT, Claude, and Gemini.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=" "
)

chunks = splitter.split_text(text)

print(chunks)