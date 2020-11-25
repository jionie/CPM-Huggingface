import torch
from transformers import GPT2LMHeadModel

from data_utils.tokenization import GPT2Tokenizer


def main():
    print("loading tokenizer")
    tokenizer = GPT2Tokenizer('bpe_3w_new/vocab.json', 'bpe_3w_new/chinese_vocab.model')
    print("loading tokenizer finished")
    src = "您好"
    input_ids = torch.tensor([tokenizer.encode(src)]).cuda()

    print("loading model")
    model = GPT2LMHeadModel.from_pretrained("model/CPM/")
    model.cuda()
    print("loading model finished")

    # generate text until the output length (which includes the context length) reaches 50
    print("testing greedy")
    greedy_output = model.generate(input_ids, max_length=50)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0].tolist()))
    print("testing greedy finished")

    # set no_repeat_ngram_size to 2
    print("testing beam search")
    beam_output = model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(beam_output[0].tolist()))
    print("testing beam search finished")

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    print("testing sampling")
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3
    )

    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output.tolist())))

    print("testing sampling finished")

    return


if __name__ == "__main__":
    main()

