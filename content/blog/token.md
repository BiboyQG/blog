+++
title = "Let's build GPT from scratch with BPE!"
date = 2024-05-08T14:15:07-05:00
draft = false
categories = ['Workshop']
tags = ['NLP', 'Byte Pair Encoding', 'Tokenization']
+++

#### 1. Workshop Description

Quick question: Have you ever thought about a string being transformed into a word vector so that it can be further fed into a machine learning algorithm?

In this workshop, we are going to dive into the fascinating world of Natural Language Processing (NLP) with our focus on Byte Pair Encoding (BPE) algorithm. We will discover how this powerful technique segments text into subword units, enabling efficient representation of words as vectors. Join us to unravel the intricacies of transforming linguistic data into numerical form, which is essential for tasks like text classification, translation, and sentiment analysis!

#### 2. Structure of the Workshop

- Introduction:
  - What is tokenization and why is it important?
- New essential tools:
  - Regex and regular expression
- "Building up" examples and discussion:
  - Tokenization related issues
  - Tokenization by example
- Major part:
  - string in Python:
    - Unicode code points -> UTF -8, UTF-16, UTF-32
  - Why not just directly use the above four methods to encode the string?
  - Introduction of BPE:
    - Why BPE -> How it works
  - Implementation of BPE:
    - Finding the most common consecutive pair.
    - Merging the most common pair to get merge rules.
    - How to train the tokenizer?
    - How to decode tokens to strings?
    - How to encode strings to tokens?
  - How can we improve from basic BPE?
    - Forced splits using regex.
- Conclusion

In the introduction segment of our workshop, we'll dive into the fundamental concept of tokenization and explore the challenges and intricacies it entails. We'll discuss how tokenization serves as the initial step in NLP tasks, highlighting its significance in breaking down raw text into manageable units for analysis.

Following the introduction, we'll introduce you to essential tools that assist in tokenization, with a special focus on Regular Expressions (Regex). Regex offers a powerful and flexible method for pattern matching and text manipulation, making it indispensable for tasks like tokenization where precise pattern recognition is crucial. And the reason we introduce it is not because we are going to use it for tokenization, we use it to achieve better performance of our tokenizer, which is going to be illustrated at the very end of our workshop.

Moving forward, we'll engage the audience in an interactive session of "building up" examples and discussions centered around tokenization. Through hands-on examples and discussions, you will gain a deeper understanding of tokenization techniques, exploring various reasons why we need a better tokenization scheme to segmenting text into meaningful units. Additionally, we'll point out the answer to some common issues such as why Large Language Models (LLM) can't spell words or why they are bad at simple Math or Python, which in this case I believe many of you have already known the answer - tokenization.

The major part of our workshop will focus on the core concepts of string manipulation in Python, starting from Unicode code points and progressing to different UTF encodings. We'll then dive into the limitations of direct encoding methods and then introduce Byte Pair Encoding (BPE) as a more efficient and effective alternative. You will learn why BPE is a preferred approach and how it operates, followed by a step-by-step implementation guide covering tokenization, training the tokenizer, and decoding/encoding strings to tokens. Additionally, we'll explore methods to enhance basic BPE, including the incorporation of regex for forced splits, ensuring more precise and effective tokenization results.

#### 3. Introduction

So our goal is to transform a string of text into numeric representation so that it can be fed directly into machine learning algorithms. By that, we need tokenization.

But what is tokenization?

Tokenization is the process of breaking down raw text into smaller, meaningful units called tokens. These tokens could be words, phrases, symbols, or any other unit of text that holds significance in understanding the whole text, meaning it serves as the foundation for all subsequent text processing tasks in NLP.

Consider a sentence: "Real integrity is doing the right thing, knowing that nobody's going to know whether you did it or not." by Oprah Winfrey. A simple tokenization of this sentence would involve breaking it down into individual words: ["Real", "integrity", "is", "doing", "the", "right", "thing", ",", "knowing", "that", "nobody", "'s", "going", "to", "know", "whether", "you", "did", "it", "or", "not", "."]. Each word here represents a token, and this tokenized representation enables machines to understand and analyze the text more effectively since it divides large chunk into smaller pieces.

However, tokenization isn't as straightforward as splitting text on spaces or punctuation marks. It actually involves addressing various linguistic challenges, such as handling contractions ("can't" should be tokenized as ["can", "'t"]), dealing with special cases like hyphenated words ("state-of-the-art" should be tokenized as ["state", "-", "of", "-", "the", "-", "art"]), and distinguishing between different types of punctuation marks, which is exactly the reason why we are going to introduce new essential tools in the next section.

Moreover, tokenization strategies may vary depending on the specific requirements of the NLP task at hand, or even specific languages. For instance, while word-level tokenization is common, there are scenarios where character-level or subword-level tokenization may be more appropriate, such as in languages with complex morphology like Chinese or when dealing with out-of-vocabulary words.

In essence, tokenization aims to transforming textual data into a format that machines can comprehend and process, bridging the gap between natural language and numerical representations which are of great siginicance for machine learning algorithms.

#### 4. New Essential Tools

In order to dedicatedly deal with contractions tokenization like `'s` or `'t`, we will need one powerful tool that enable us to do different text pattern matching, which is Regex. This is an extension of the original re module inside original Python, meaning it is more powerful.

And in our workshop, we will focus on its two specific function calls and one specific regular expression referenced from [this](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/simple_tokenizer.py#L78) repository with some modification:

```python
# import modules and rename it as re for convenience
import regex as re

# define pattern referenced from the repo except for special tokens
pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.IGNORECASE)

text = 'Any text that you want to be split.'
print(re.findall(pat, text))
```

We will first walkthrough `re.findall` and then dive into `re.compile` and the meaning of the regular expression inside it.

- `re.findall`:

  - Basically what `re.findall` does is it will take the regular expression pattern, in this case `pat` and try to match it against the `text`. The way it works is that the `regex` module will go through the string from left to right to match the pattern, and `re.findall` will match all the occurences and organize them into a list. In this case, the output of the terminal will be:
    ```shell
    ['Any', ' text', ' that', ' you', ' want', ' to', ' be', ' split', '.']
    ```

- `re.compile`:

  - On the other hand, what `re.compile` does is defining a regular expression that can be used for further regex function using the regular expression string inside it.

- Regular expression:

  - We will lay our main focus on the regular expression here from two perspectives: what it means and why should it be like this.

    - What the regular expression means:

      - `'s|'t|'re|'ve|'m|'ll|'d`:

        - The `|` inside the regular expression represents the "or" operation in regex, meaning that it will match the pattern either on its left hand side or on its right hand side. And therefore we understand the this part of regular expression, which matches common English contractions and possessive endings.
          For example, for text:

          ```python
          text = "Any text that you'd want to be split."
          print(re.findall(pat, text))
          ```

          The output will be:

          ```shell
          ['Any', ' text', ' that', ' you', "'d", ' want', ' to', ' be', ' split', '.']
          ```

          In this case we can clearly see that `'d'` is successfully separated out.

      - `  ?\p{L}+`:

        - For this chunk we can take a look at [this](https://www.regular-expressions.info/unicode.html) source. Firstly, ` ?` optionally matches a space before the main pattern. Next, according to the source, `\p{L}` is a Unicode property escape that matches any kind of letter from any language. And `+` means one or more of the preceding element, so `\p{L}+` matches one or more letters (i.e., it matches the whole words). Therefore, this chunk match the whole word with an optional space in front of it.

      - `  ?\p{N}+`:

        - Similar to the previous explanation, this matches an optional leading space followed by one or more Unicode numeric characters (i.e., it matches entire numbers).
          For example, for text:

          ```python
          text = "Any text123 that you'd want to be split."
          print(re.findall(pat, text))
          ```

          The output will be:

          ```shell
          ['Any', ' text', '123', ' that', ' you', "'d", ' want', ' to', ' be', ' split', '.']
          ```

          In this case we can clearly see that the number is separated out.

      - `?[^\s\p{L}\p{N}]+`:

        - `  ?` again optionally matches a space.

        - `^` represents "not" operation in regex regular expression. Therefore, `[^\s\p{L}\p{N}]` matches any character that is not a whitespace (`\s`), not a letter (`\p{L}`), and not a numeric character (`\p{N}`), therefore is used to match punctuation and other symbols.

        - `+` ensures that one or more of such characters are matched together as a group, in this case it matches as many punctuations as possible.
          For example, for text:

          ```python
          text = "Any text that you'd want to be split???!!!"
          print(re.findall(pat, text))
          ```

          The output will be:

          ```shell
          ['Any', ' text', ' that', ' you', "'d", ' want', ' to', ' be', ' split', '???!!!']
          ```

          In this case we can clearly see that the punctuations are captured.

      - `\s+(?!\S)`:

        - `\s+` matches one or more whitespace characters here.

        - `(?!\S)` is a negative lookahead that asserts what follows is not a non-whitespace character. This part will ensure that the matched whitespace is at the end of a line or string, preventing excessive spaces from consuming lots of spaces at the same time.
          For example, for text:

          ```python
          text = "Any text that you'd want to                 be split."
          print(re.findall(pat, text))
          ```

          The output will be:

          ```shell
          ['Any', ' text', ' that', ' you', "'d", ' want', ' to', '                ', ' be', ' split', '.']
          ```

          In this case we can clearly see that the excessive space is successfully separated out with one space in front of "be".

      - `\s+`:

        - This simply matches one or more whitespace characters anywhere else in the input like the one above.

    - Why should it be like this?

      - The reason is actually explained above, but the main reason is that this regular expression has the ability to match the whole word, whole integer, punctuation(s) and excessive spaces, making it really useful in the final refinement process. This will be further discussed in the final section.

#### 5. "Building up" examples and discussion

In this section, we will dive deeper into the practical aspects of tokenization, exploring both common challenges through illustrative examples and engaging discussions.

A quick fun and useful app that we can utilize here is the [tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt2) website, where it visualizes the tokenization live just in the browser. Since GPT-2 uses BPE algorithm with minimal modification, we are going to use its tokenizer as an example.

![image-20240506173131577](https://s2.loli.net/2024/05/08/U1WmXwK36TAS5b7.png)

Here we can see the visualization of the tokenization. For the english sentence (selected from IS 305 [website](https://elliewix.gitbook.io/is-305/module-6-semi-structured-data-to-tables)), it is pretty clear that BPE algorithm tokenizes the sentence almost perfectly (with space being part of the token chunk) according to its merge rules, which seems all well and good. But this exactly the reason why LLMs are bad at spelling! Since nearly all of the common words are tokenized so that they can be represented using only one single integer, it's really hard for them to spell the characters within that single token which are treated as a single integer.

What's more, in terms of some simple arithmetics, we can see that the tokenizer handle the first equation perfectly, but when it comes to the second one, the two number, 3221 and 2334, are fed into the machine learning as two separate tokens respectively (3, 221 and 233, 4). Even the final result 5555 is divided into 5 and 555, which to some extent explains why sometimes LLMs are bad at simple arithmetics -> tokenization!

And another example is that we write our egg under multiple context. One is `Egg.` , where the BPE separates the period from the noun. In the sentence, the Egg is separated together with the space in front of it. When we write `egg.`, the "egg" is separated from period, but when we write `EGG.`, the tokenized version is like `['EG', 'G', ''.']`. Here we notice that, for the same concept, "egg", depending on its location within the sentence and upper or lowercase or mixed, all these will result in fairly different tokens and therefore different word vector, which will be further learned by the machine learning algorithms.

Additionally, the third example is quite interestring. Here, the text `你好，我是池邦豪！ 👋 ` means 'Hello, my name is Banghao Chi' in English. By punctuation, I believe that everybody can guess the meaning of "你好", right? That's correct! "你好" exactly means "Hello" in English. But in this case, what we observe is that BPE divides "你好" into two seperated tokens, which is really weird since if "你好" means "Hello", why BPE would divide them apart? Here we can notice that BPE works slightly worse on non-english languages, part of reason is because the training dataset for BPE on English is much larger than any other languages, resulting in different tokenization performances on different languages.

Finally, we come to a snippet of Python code here. What we can notice is that the indent here are all treated as individual space and all individual space are all separate tokens, which leads to extremely wasteful tokenization in this way. In better tokenizer (taking the second line of code as an example), we will combine the first three spaces together into one single token and then treat `  if` as the other token. This example also to some extent explains why LLMs sometimes can perform really bad at coding -> tokenization!

After coming a long way from scratch, we now understand the significance of the tokenizer, and in the following sections, we are going teach all the fundations of the BPE and why we should choose it. And based on that, you can then explore the fascinating world of tokenization and therefore even solve the issues above on your own!

#### 6. What encoding mechanism should we choose?

One thing we need to remember is that we not only want to support English but some other popular languages and also some special characters like emoji. Then how are we going to feed our text into the tokenizer?

If we take a quick look at the official Python documentation, we can see that:

> Strings are immutable [sequences](https://docs.python.org/3/library/stdtypes.html#typesseq) of Unicode code points.

![image-20240428210054360](https://s2.loli.net/2024/05/08/6yYpanFXNirCbMG.png)

But what are Unicode code points ? It can be found [here](https://en.wikipedia.org/wiki/Unicode) in Wikipedia. So basically it is a definition of roughly 150,000 characters defined by Unicode Consortium as part of the Unicode standard. Roughly speaking, they define how each character look like and what integers represent them. If we scroll to the bottom, we can even see that it's very much alive, still updated each year.

![image-20240507083257417](https://s2.loli.net/2024/05/08/tOcI6lKkyMJCiba.png)

The way we can access the Unicode code points integer in Python is through `ord` function.

```python
[ord(t)for t in "你好，我叫池邦豪 👋 (hello, my name is Banghao Chi in Chinese)"]
```

And we can find the outputs of the terminal being:

```shell
[20320,  22909,  65292,  25105,  21483,  27744,  37030,  35946,  32,  128075,  32,  40,  104,  101,  108,  108,  111,  44,  32,  109,  121,  32,  110,  97,  109,  101,  32,  105,  115,  32,  66,  97,  110,  103,  104,  97,  111,  32,  67,  104,  105,  32,  105,  110,  32,  67,  104,  105,  110,  101,  115,  101,  41]
```

But you might ask, we already turned the raw code points into integer format, so why can't we just simply use these integers and not doing any tokenization at all?

Well there are three main reasons:

- Unicode has a vocab size which is too large (roughly 150,000) for NLP or any other machine learning algorithms.
- It is consistently updated, which is not a stable representation that we may want to use.
- We don't want to achieve character-level encoding due to the nature of English.

Because of these, we may need something better, turning to encodings. If we scroll a little bit down of the wikipedia page of Unicode, we can find that the Unicode Consortium has defined three extra types of encodings by which we can take Unicode text and transform them into binary data:

- UTF-8 (has 256 characters)
- UTF-16
- UTF-32

Among all of them, UTF-8 is the most common one, and as we can see from its wikipedia page, it takes every single code point and it translates it to a byte stream which is between one to four bytes:

![image-20240507103245941](https://s2.loli.net/2024/05/08/xCZ76E2inVWu3AP.png)

So if we'd like to do the UTF encoding, how can we do that in Python? Well, string object has a method `encode` and we can choose what encoding scheme we want inside it.

```python
"你好，我叫池邦豪 👋 (hello, my name is Banghao Chi in Chinese)".encode("utf-8")
```

which outputs:

```shell
b'\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe6\x88\x91\xe5\x8f\xab\xe6\xb1\xa0\xe9\x82\xa6\xe8\xb1\xaa\xf0\x9f\x91\x8b (hello, my name is Banghao Chi in Chinese)'
```

But what we directly get is not very user-friendly since since they are bytes object, so instead we will convert it to a list which consists of the raw bytes of this encoding.

```python
list("你好，我叫池邦豪 👋 (hello, my name is Banghao Chi in Chinese)".encode("utf-8"))
```

And the output shows:

```shell
[228,  189,  160,  229,  165,  189,  239,  188,  140,  230,  136,  145,  229,  143,  171,  230,  177,  160,  233,  130,  166,  232,  177,  170,  32,  240,  159,  145,  139,  32,  40,  104,  101,  108,  108,  111,  44,  32,  109,  121,  32,  110,  97,  109,  101,  32,  105,  115,  32,  66,  97,  110,  103,  104,  97,  111,  32,  67,  104,  105,  32,  105,  110,  32,  67,  104,  105,  110,  101,  115,  101,  41]
```

So this is the raw bytes representing the above string accoring to the UTF-8 encoding. But what about UTF-16 and UTF-32? And how are we going to choose among three of them? To make a decision, we can just print them out to see the differences:

- UTF-8 (has 256 characters)

  ```shell
  [228,  189,  160,  229,  165,  189,  239,  188,  140,  230,  136,  145,  229,  143,  171,  230,  177,  160,  233,  130,  166,  232,  177,  170,  32,  240,  159,  145,  139,  32,  40,  104,  101,  108,  108,  111,  44,  32,  109,  121,  32,  110,  97,  109,  101,  32,  105,  115,  32,  66,  97,  110,  103,  104,  97,  111,  32,  67,  104,  105,  32,  105,  110,  32,  67,  104,  105,  110,  101,  115,  101,  41]
  ```

- UTF-16

  ```shell
  [255,  254,  96,  79,  125,  89,  12,  255,  17,  98,  235,  83,  96,  108,  166,  144,  106,  140,  32,  0,  61,  216,  75,  220,  32,  0,  40,  0,  104,  0,  101,  0,  108,  0,  108,  0,  111,  0,  44,  0,  32,  0,  109,  0,  121,  0,  32,  0,  110,  0,  97,  0,  109,  0,  101,  0,  32,  0,  105,  0,  115,  0,  32,  0,  66,  0,  97,  0,  110,  0,  103,  0,  104,  0,  97,  0,  111,  0,  32,  0,  67,  0,  104,  0,  105,  0,  32,  0,  105,  0,  110,  0,  32,  0,  67,  0,  104,  0,  105,  0,  110,  0,  101,  0,  115,  0,  101,  0,  41,  0]
  ```

- UTF-32
  ```shell
  [255,  254,  0,  0,  96,  79,  0,  0,  125,  89,  0,  0,  12,  255,  0,  0,  17,  98,  0,  0,  235,  83,  0,  0,  96,  108,  0,  0,  166,  144,  0,  0,  106,  140,  0,  0,  32,  0,  0,  0,  75,  244,  1,  0,  32,  0,  0,  0,  40,  0,  0,  0,  104,  0,  0,  0,  101,  0,  0,  0,  108,  0,  0,  0,  108,  0,  0,  0,  111,  0,  0,  0,  44,  0,  0,  0,  32,  0,  0,  0,  109,  0,  0,  0,  121,  0,  0,  0,  32,  0,  0,  0,  110,  0,  0,  0,  97,  0,  0,  0,  109,  0,  0,  0,  101,  0,  0,  0,  32,  0,  0,  0,  105,  0,  0,  0,  115,  0,  0,  0,  32,  0,  0,  0,  66,  0,  0,  0,  97,  0,  0,  0,  110,  0,  0,  0,  103,  0,  0,  0,  104,  0,  0,  0,  97,  0,  0,  0,  111,  0,  0,  0,  32,  0,  0,  0,  67,  0,  0,  0,  104,  0,  0,  0,  105,  0,  0,  0,  32,  0,  0,  0,  105,  0,  0,  0,  110,  0,  0,  0,  32,  0,  0,  0,  67,  0,  0,  0,  104,  0,  0,  0,  105,  0,  0,  0,  110,  0,  0,  0,  101,  0,  0,  0,  115,  0,  0,  0,  101,  0,  0,  0,  41,  0,  0,  0]
  ```

From the above outputs, we start to see the disadvantages of UTF-16 and UTF-32, with excessive 0 lying within the list, which give us a sense that UTF-16 and UTF-32 are a bit of a wasteful encodings. Therefore, we will stick with UTF-8 for better efficiency.

However, we won't just barely use UTF-8 encoding either, since it only has a vocabulary size of 256, meaning that we will only have 256 types of tokens, which is too small. The small size of vocabulary will results in the huge length representation of the original text, which is not nice for the up-coming machine learning algorithms. Therefore, we need to find a way to increase our vocabulary size by compressing the original bytes sequences to reduce the final representation length of the text, which turns out to be BPE.

#### 7. Introduction of BPE

The process of BPE is quite intuitive, according to [Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding), but we will walk you through to help you get the basic idea of the algorithm. The following example is referenced from [Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding):

Suppose the data to be encoded is:

```shell
aaabdaaabac
```

The byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, such as "Z". Now there is the following data and replacement table:

```tex
ZabdZabac
Z=aa
```

Then the process is repeated with byte pair "ab", replacing it with "Y":

```tex
ZYdZYac
Y=ab
Z=aa
```

The only literal byte pair left occurs only once, and the encoding might stop here. Alternatively, the process could continue with [recursive](https://en.wikipedia.org/wiki/Recursion) byte pair encoding, replacing "ZY" with "X":

```tex
XdXac
X=ZY
Y=ab
Z=aa
```

This data cannot be compressed further by byte pair encoding because there are no pairs of bytes that occur more than once.

So basically after we've gone through the process, instead of having a sequence of 11 tokens with a vocabulary size of 4, we now have a suquence of 5 tokens with a vocabulary size of 7. By using BPE, we can iteratively compress our original sequence with more vocabulary and therefore find a balance of tokenization for further use.

#### 8. Implementation of BPE

Since we need some training texts to train our BPE, we first select some texts from the [IS 305](https://elliewix.gitbook.io/is-305) website and do some cleaning.

```python
corpus = ...
corpus = corpus.replace('\n', ' ')
corpus = corpus.replace('    ', ' ')
corpus = corpus.replace('   ', ' ')
corpus = corpus.replace('  ', ' ')
```

Then we convert the corpus to raw bytes, but in order to manipulate them in a more convenvient way, we use the `map` function to convert the bytes object into integer and store them in a list.

```python
encoding = 'utf-8'
tokens = corpus.encode(encoding)
tokens = list(map(int, tokens))
print("length of corpus:", len(corpus))
print('----------------')
print("length: tokens", len(tokens))
print('----------------')
print("Length equal: ", len(corpus) == len(tokens))
```

The outputs show:

```shell
length of corpus:  29678
----------------
length of tokens:  29681
----------------
Length equal:  False
```

The reason why of length of them is not equal here is because although most of the characters are simple ASCII characters which just become a single byte or integer after encoding, but for some Unicode, more complex characters like emoji, they become multiple bytes up to 4 instead of just 1.

**Count the Frequency**

In order to find the pair of bytes that occur most frequently, we first need to count the frequency of all pairs because them we are going to merge them based on their frequency.

```python
def count(tokens):
    counts = {}
    for p in zip(tokens, tokens[1:]):
        counts[p] = counts.get(p, 0) + 1
    return counts

counts = count(tokens)
```

So what we do is that we pass the tokens (a list of integer) as the input and we create a new dictionary to store the frequency of each pair. We then iterate through adjacent elements in the tokens. If this is the first time we come across this pair, the value returned from `counts.get(p, 0)` will be zero and we add one to show that this is the first time. Then we use the pair (which is a tuple of consecutive elements in tokens). The case where it's not the first time is trivial, we just get the frequency from `counts.get(p, 0)` and add one to it. After the process, we return the dictionary which store the frequency of each pair.

The outputs of counts should be something like this:

```tex
{(240, 159): 1,
 (159, 145): 1,
 (145, 139): 1,
 (139, 32): 1,
 (32, 77): 14,
 (77, 111): 7,
 (111, 100): 61,
 (100, 117): 17,
 (117, 108): 97,
 (108, 101): 266,
 (101, 32): 939,
 ...}
```

We can notice that the combination of `(101, 32)` has the highest frequency, and we can reverse to see what combination of them is by using the `chr` function:

```python
print(chr(101), chr(32))
```

The outputs are:

```shell
('e', ' ')
```

The `chr` function is just a reverse function of `ord`. The outputs mean that the most common pair in the text is `('e', ' ')`.

**Get Top Pair**

After getting the frequency of each adjacent pairs, we should get the pair which has the highest frequency through `max` function instead of telling by bare eyes:

```python
top_pair = max(counts, key=counts.get)
```

By defining the key to be the frequency, we can find the top pair and name it `top_pair`.

**Merge pair functionality based on a specific pair we got**

We then need to replace all consecutive occurences of pair with the new idx:

```python
def merge(tokens, pair, idx):
    # create new list to return later
    new_tokens = []
    # reset pointer
    i = 0
    # we iterate through every token
    while i < len(tokens):
        # pair match for first two and ensure not the last token for the last condition
        if  i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(idx)
            # increase by 2 to skip
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens
```

We pass `tokens`, `pair`(top_pair), `idx`(integer that we want to represent the pair) to the function as the inputs. We will create a new tokens list for convenience and set our pointer `i` to zero meaning the starting index. Then we will iterate through the whole `tokens` to merge the tokens within the list based on `pair` using `while i < len(tokens):`.

Inside the `while` loop, we first see if the current token is the last one:

- if so, we go straight to the `else` branch and append it to the `new_tokens`;
- if not, then need to verify if the current token and the next one match the pair:
  - if so, we append the new `idx` to the `new_tokens` and add `i` by 2 to skip to thee third token of the current sequence.
  - if not, we just append the current token to the `new_tokens` and go to the next token.

Hence, we get the above merge function, and in order to test if it is functioning correctly, we do a simple test as follows:

```python
# a simple test
print(merge([2, 2, 8, 9, 5, 3, 8, 9], (8, 9), 257))
```

The outputs show:

```tex
[2, 2, 257, 5, 3, 257]
```

which means our function is correct.

**Get Merge Rules**

We can then define the size of the vocab that we want and then integrate all the code for now to get the merge rules which are later used in the encoding stage:

```python
vocab_size = 400 # the final vocab length we want
num_merges = vocab_size - 256 # UTF has original characters, so here we minus it
tokens_BPE = list(tokens)
```

Here `vocab_size` is just a hyper-parameter, meaning that this is the parameter that we can set whatever we want, representing the final size of the vocabulary. We can then get the number of merges by substracting it with 256 (UTF-8 has 256 characters originally, so we substract it). In case of mutating the orignal `tokens` list, we copy a new version of it and name it `tokens_BPE`.

After getting the `num_merges`, `tokens_BPE`, we can perform the merge operation and get the merge rules:

```python
merges = {} # key: pair to be merged; value: new token idx
for i in range(num_merges):
    counts = count(tokens_BPE)
    pair = max(counts, key=counts.get)
    idx = 256 + i
    print(f"merging {pair} into a new idx {idx}")
    tokens_BPE = merge(tokens_BPE, pair, idx)
    merges[pair] = idx
```

Here we create a new dictionary `merges` to store the merge rules. Then we iterate `num_merges` of times to:

1. First get the frequency of all pairs.

   ```python
   counts = count(tokens_BPE)
   ```

2. Find the top pair based on the first step and the frequency. Here, `key=counts.get` automatically tells the program that to find the pair which has the maximum value (frequency).

   ```python
   pair = max(counts, key=counts.get)
   ```

3. Get the new idx which we want to represent the new top pair.

   ```python
   idx = 256 + i
   ```

4. Use `merge` function to merge the pairs within the `tokens_BPE`.

   ```python
   tokens_BPE = merge(tokens_BPE, pair, idx)
   ```

5. Save this merge rule to the merge rules.
   ```python
   merges[pair] = idx
   ```

After executing the cell, we can get the outputs:

```tex
merging (101, 32) into a new idx 256
merging (116, 104) into a new idx 257
merging (115, 32) into a new idx 258
merging (116, 32) into a new idx 259
merging (105, 110) into a new idx 260
merging (97, 110) into a new idx 261
merging (101, 114) into a new idx 262
merging (32, 257) into a new idx 263
merging (111, 110) into a new idx 264
merging (97, 116) into a new idx 265
...
```

**Get Vocabulary List**

After that, we can get the completed vocabulary list based on merge rules, which is going to be used in decoding stage:

```python
vocab = {idx: bytes([idx]) for idx in range(256)} # original characters in UTF
# then we pop the pair out to vocab pair by pair
for (pair0, pair1), idx in merges.items():
    vocab[idx] = vocab[pair0] + vocab[pair1]
```

For the first line, we place the original 256 characters defined by UTF-8 inside the `vocab` variable. The key here is the idx and the value is the bytes object of that idx.

Then, we append the contatenation of the pair inside the merge rules to the `vocab` so that it not only contains the 256 characters defined by UTF-8, but also contains the pairs inside the merge rules.

To better help you understand the code, I attach the content of `merges` and `vocab` below:

- `merges`:

  ```tex
  {(101, 32): 256,
   (116, 104): 257,
   (115, 32): 258,
   (116, 32): 259,
   (105, 110): 260,
   (97, 110): 261,
   (101, 114): 262,
   (32, 257): 263,
   (111, 110): 264,
   ...}
  ```

- `vocab`:

  ```shell
  {0: b'\x00',
   1: b'\x01',
   2: b'\x02',
   3: b'\x03',
   4: b'\x04',
   5: b'\x05',
   6: b'\x06',
   7: b'\x07',
   8: b'\x08',
   ...
   382: b'with ',
   383: b'pro',
   384: b'comm',
   385: b"', '",
   386: b'resul',
   387: b'on ',
   388: b'will ',
   389: b'ver',
   390: b'if',
   391: b'use ',
   392: b'pl',
   393: b'ob',
   394: b'al ',
   395: b'ent',
   396: b'ut ',
   397: b'files ',
   398: b'sp',
   399: b'obj'}
  ```

​ Through `vocab` we can already see that some of the most common word are already grouped up, such as "with", "will", "use", "if", "files", even some of the prefix or suffix such as "pro", "comm", etc. As long as we increase the `vocab_size`, more common words will be in the `vocab` and `merges`.

**Decoding Stage**

For decoding stage, what we get as the input is a list of integers that represents the encoded text, and what we return in this function will be the Python string. Therefore:

```python
def decode(tokens_BPE):
    tokens = b"".join(vocab[idx] for idx in tokens_BPE)
    text = tokens.decode(encoding, errors="replace")
    return text
```

We will also walk you through line by line in this case:

1. We get the original bytes representation of the text by first using the `idx` and `vocab` to look up the bytes, and contatenating all of them.

   ```python
   tokens = b"".join(vocab[idx] for idx in tokens_BPE)
   ```

2. After we get the raw bytes, we will decode them using `str.decode` method. This is because previously we use `encode` method to generate raw bytes using python string, and in order to reverse the process, we will be using the `decode` method.

   ```python
   text = tokens.decode(encoding, errors="replace")
   ```

   The reason why we need a `errors="replace"` is because, there is a certain scheme that the UTF-8 do the conversion between code point and byte.
   ![image-20240507103245941](https://s2.loli.net/2024/05/08/xCZ76E2inVWu3AP.png)
   For example, if we try the following code without `error='replace'`:

   ```python
   def decode(tokens_BPE):
       tokens = b"".join(vocab[idx] for idx in tokens_BPE)
       text = tokens.decode(encoding)
       return text

   decode([128])
   ```

   The output shows:

   ```tex
   UnicodeDetectError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
   ```

   This is because the binary representation of 128 starts with 1 and all others are 0, meaning that it doesn't match any of the above scheme, explaining why it can't be decoded. Therefore, the way we fix this is add the keyword argument `error='replace'` so that the outputs will become:

   ```tex
   '�'
   ```

   according to the offical Python documentation:

   ![image-20240507164541595](https://s2.loli.net/2024/05/08/7Z8IaV2YcfTE6SP.png)

**Encoding Stage**

Now we are going in the other way - implementing the encoding stage. What we now get as the input will be the string, and we want to convert them into bytes. Therefore:

```python
def encode(corpus):
    # we first get the tokens list and convert them
    tokens = list(corpus.encode(encoding))
    # in case of single character or blank; that way, no need to merge anything
    while len(tokens) > 1:
        counts = count(tokens)
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            # nothing more can be merged
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
```

We will also walk you through line by line in this case:

1. Firstly, we get the bytes representation of the original string.

   ```python
   tokens = list(corpus.encode(encoding))
   ```

2. Next, we only begin the merge process when the length of our `tokens` list is greater than 1; otherwise, no need to merge anything, since there's only 1 character.

   ```python
   while len(tokens) > 1:
   ```

3. Inside the loop, to reuse some of the code, we use `count` to get the pair within the original string.

   ```python
   counts = count(tokens)
   ```

4. We are going to merge the pair which has the smallest idx within `merges`. This is because, as we take a look at `merges`:

   ```tex
   {(101, 32): 256,
    (116, 104): 257,
    (115, 32): 258,
    (116, 32): 259,
    (105, 110): 260,
    (97, 110): 261,
    (101, 114): 262,
    (32, 257): 263,
    (111, 110): 264,
    ...}
   ```

   We can clearly see that some of the later-on pair depends on the previous merge, for example, `(32, 257): 263` depends on 257. Therefore, we need to select the pair that has the minimal idx to become the pair that we want to merge in each iteration.

   In Python, if we call `min` on an iterator, in this case, a dictionary, we'll be iterating the keys of it. And here we need to use a lambda function since we are going to use some other variable besides `counts`.

   Therefore, `pair = min(counts, key=lambda p: merges.get(p, float("inf")))` in this case, will return us the pair in `counts` which has the lowest idx within `merges` (if it's not in `merges`, we return infinity so that it won't be one of the candidates).

   ```python
   pair = min(counts, key=lambda p: merges.get(p, float("inf")))
   ```

5. After getting the pair we want to merge, we need to see if it's in `merges`:

   - If not, it means that there's no pair left to be merged, so we break out of the loop;
   - If so, we continue.

   ```python
   if pair not in merges:
       break # nothing more can be merged
   ```

6. If there's pair we need to merge, since our goal is to encode the string, we then first need to get the corresponding idx of the pair.

   ```python
   idx = merges[pair]
   ```

   Then, by calling `merge` function, we complete one iteration of the encoding stage.

   ```python
   tokens = merge(tokens, pair, idx)
   ```

7. Finally, if we are out of the loop, we return the `tokens`, which is the bytes representation of the original text.
   ```python
   return tokens
   ```

**Verify our tokenizer**

To verify if our tokenizer is working, we can simply encode a text and then decode it to see if the final output is the same as the original one.

```python
print(encode("Hello👋, my name is Banghao Chi!; 你好👋，我是池邦豪！"))
```

The outputs of it is:

```tex
[72, 351, 353, 240, 159, 145, 139, 270, 109, 273, 110, 334, 256, 287, 66, 261, 103, 340, 329, 67, 104, 105, 33, 59, 32, 228, 189, 160, 229, 165, 189, 240, 159, 145, 139, 239, 188, 140, 230, 136, 145, 230, 152, 175, 230, 177, 160, 233, 130, 166, 232, 177, 170, 239, 188, 129]
```

We then store it in a variable:

```python
tokens_TEST = [72, 351, 353, 240, 159, 145, 139, 270, 109, 273, 110, 334, 256, 287, 66, 261, 103, 340, 329, 67, 104, 105, 33, 59, 32, 228, 189, 160, 229, 165, 189, 240, 159, 145, 139, 239, 188, 140, 230, 136, 145, 230, 152, 175, 230, 177, 160, 233, 130, 166, 232, 177, 170, 239, 188, 129]
```

And we decode it to see the final outputs:

```python
print(decode(tokens_TEST))
```

which outputs:

```tex
Hello👋, my name is Banghao Chi!; 你好👋，我是池邦豪！
```

meaning that our tokenization is working as expected!

#### 9. How can we improve from here? (Hint)

We've saw how we can take some training on raw texts using BPE algorithm. The parameters of it is just the merge rules. Once we have the merge rules, we can both encode and decode between raw texts and token sequences.

What we now are going to do is that we are going to take a look at some papers talking about the tokenizer. For example, in page 4 of [this](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) paper, we see that BPE ends up with `dog`, `dog.`, `dog!`, `dog?`, something like these with a simple concept. It feels like we are clustering things that shouldn't be clustered, and we are combining semantics with punctuation, which is suboptimal.

![image-20240507174730610](https://s2.loli.net/2024/05/08/oVlYbwyS7PveCWT.png)

Hence, the researcher in the paper points out a manual way of enforcing some types of the characters should never be merged together, which turns out to be the tool that we introduce at the beginning of the workshop!

```python
# import modules and rename it as re for convenience
import regex as re

# define pattern referenced from the repo except for special tokens
pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.IGNORECASE)

text = 'Any text that you want to be split.'
print(re.findall(pat, text))

# outputs:
# ['Any', ' text', ' that', ' you', ' want', ' to', ' be', ' split', '.']
```

By forced splitting the original text, we therefore ensure that we will never merge between any adjacent strings within the list. After we've done the merging for all of these strings individually, we just simply concatenate them to generate the final representation, which is pretty similar for the training and decoding stage for the BPE.

#### 10. Conclusion

In this workshop, we explored the transformative process of tokenization, particularly focusing on the Byte Pair Encoding (BPE) algorithm. We dived into the intricacies of converting textual data into a numerical format, which is essential for various post machine learning applications such as text classification, sentiment analysis, and language translation.

In conclusion, we first began with an introduction to the fundamentals of tokenization and its significance in machine learning. Understanding that tokenization is more than merely splitting text into words, we underscored its role in handling linguistic challenges it presents, such as managing contractions and special characters.

Through practical demonstrations and discussions, we demonstrated a deeper understanding of tokenization, Python's string manipulations, emphasizing Unicode and UTF encodings. We then introduced BPE, explaining its efficiency in reducing vocabulary size while maintaining meaningful linguistic units. Our hands-on sessions included implementing BPE from scratch, illustrating the process of finding and merging the most common byte pairs in a corpus.

Furthermore, we explored how enhancements, such as using regex for forced splits, could refine the BPE process, ensuring that semantically different units like punctuation and words are not inappropriately merged. This approach helps in fine-tuning tokenizers to handle diverse linguistic patterns more effectively.

As we wrapped up, we demonstrated the practical application and the potential improvements in tokenizer designs. By understanding and implementing these advanced techniques, participants are now better equipped to tackle complex tasks and contribute to the evolution of more effective machine learning algorithms such as making LLMs better at spelling, Math and Python.

In conclusion, we not only provided a comprehensive overview of BPE and its applications but also emphasized the reason why we should choose it and the continuous need for innovation within this field. The knowledge and skills gained here are steps towards mastering the art and science of future exploration, paving the way for advancements in data manipulation and even machine learning!
