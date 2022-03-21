from transformers import T5TokenizerFast

from packages.choice.model import T5PromptTuningForConditionalGeneration
from packages.choice.standalone import Standalone

from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    path_to_model = 'packages/choice/ipoie'
    max_step = 10
    device = -1
    query = 'lion'

    tokenizer = T5TokenizerFast.from_pretrained(path_to_model)
    model = T5PromptTuningForConditionalGeneration.from_pretrained(path_to_model)

    standalone = Standalone(model=model, tokenizer=tokenizer, max_step=max_step, device=device)
    extraction = standalone.pipeline([query], batch_size=32)

    res = map(lambda x: (x[0], x[1]), extraction[query].items())
    print(list(res))

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
