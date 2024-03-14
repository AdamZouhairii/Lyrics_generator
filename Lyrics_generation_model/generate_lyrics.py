import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class LyricsGenerator:
    def __init__(self, model_path):
        """
        Initializes the LyricsGenerator class.

        Args:
            model_path (str): The path to the pre-trained GPT-2 model.
        """
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def preprocess_input(self, text):
        """
        Preprocesses the input text by encoding it using the GPT-2 tokenizer.

        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: The input text encoded as tensor.
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        return input_ids

    def generate_output(self, input_ids, max_length, num_return_sequences, temperature, pad_token_id, top_k, top_p, repetition_penalty, do_sample, use_cache):
        """
        Generates the output sequences using the pre-trained GPT-2 model.

        Args:
            input_ids (torch.Tensor): The input text encoded as tensor.
            max_length (int): The maximum length of the generated sequences.
            num_return_sequences (int): The number of sequences to generate.
            temperature (float): The temperature value for controlling randomness in generation.
            pad_token_id (int): The token ID for padding.
            top_k (int): The number of highest probability tokens to consider for top-k sampling.
            top_p (float): The cumulative probability threshold for top-p nucleus sampling.
            repetition_penalty (float): The penalty for repeating tokens.
            do_sample (bool): Whether to use sampling during generation.
            use_cache (bool): Whether to use the cache during generation.

        Returns:
            torch.Tensor: The generated output sequences.
        """
        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            pad_token_id=pad_token_id,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        return output_sequences

    def postprocess_output(self, sequences):
        """
        Postprocesses the generated output sequences by decoding them using the GPT-2 tokenizer.

        Args:
            sequences (torch.Tensor): The generated output sequences.

        Returns:
            list: The postprocessed generated texts.
        """
        generated_texts = []
        for sequence in sequences:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(text)
        return generated_texts

    def generate_lyrics(self, category, max_length=1000, num_return_sequences=1, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.0, do_sample=True, use_cache=True):
        """
        Generates lyrics based on the given category.

        Args:
            category (str): The category for generating lyrics.
            max_length (int): The maximum length of the generated lyrics.
            num_return_sequences (int): The number of lyrics to generate.
            temperature (float): The temperature value for controlling randomness in generation.
            top_k (int): The number of highest probability tokens to consider for top-k sampling.
            top_p (float): The cumulative probability threshold for top-p nucleus sampling.
            repetition_penalty (float): The penalty for repeating tokens.
            do_sample (bool): Whether to use sampling during generation.
            use_cache (bool): Whether to use the cache during generation.

        Returns:
            list: The generated lyrics.
        """
        try:
            with torch.no_grad():
                input_text = f"{category}: "
                input_ids = self.preprocess_input(input_text)
                output_sequences = self.generate_output(input_ids, max_length, num_return_sequences, temperature, self.tokenizer.eos_token_id, top_k, top_p, repetition_penalty, do_sample, use_cache)
                generated_lyrics = self.postprocess_output(output_sequences)
                return generated_lyrics
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            return []

# Usage
model_path = r"\gpt2-small-chatlyrics"
category = "Generate Rock\nPost-Rock\nTrip-Hop\nElectronic Rock\nExperimental Rock\nBritish Rock\nArt Pop\nUK\nAlternative Rock\nArt Rock type lyrics  like Radiohead"

generator = LyricsGenerator(model_path)
generated_lyrics = generator.generate_lyrics(category)

print("Generated Lyrics:")
for lyrics in generated_lyrics:
    print(lyrics)