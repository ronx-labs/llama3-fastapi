# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire
import time
import datetime

from llama import Dialog, Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 8192,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    query_string = "How does Zane Huffman view the trading and utility of points in crypto markets?"

    prompt_docs = [
        "Talk Title: Actually, Points Are Cool and Good | Zane Huffman - Demos\nSpeaker: Zane Huffman\nText: Hey, everyone. Congratulations. We are at, oh, I'm going to fix that real quick. There we go. We are at the most exciting topic of the day, points. I am an avid proponent for points. My name is Zane Huffman. I give out points. I believe that by the end of this presentation, you too will feel similarly to me. If you're a point skeptic, keep an open mind. We can have some Q&A at the end. Let's dive in. All right, so today, I will give a rundown of what exactly points are. I will contextualize points in the history of crypto micro work. I will talk about why points are an awesome innovation in the space today, address some of the common critiques, things we can do to improve the future of points, and most importantly, give me your attention until the end of this presentation, and perhaps you too, audience, can earn some points yourself. So, a bit about me. My name is Zane Huffman. I go by Green Jeff Online. That's my m'lady. I have been doing stuff in this space, doing tasks and getting points, getting tokens for a long time. This is me referring people to a Bitcoin Minecraft casino in 2013. And today I also give out points. I'm the CEO at Demos, which offers orb-free human authentication, and I contribute to BLOQ, where we have some exciting upcoming projects.",
        "Talk Title: Actually, Points Are Cool and Good | Zane Huffman - Demos\nSpeaker: Zane Huffman\nText: All of these which have points. So, at a high level, points are a mechanic that businesses can award users, award their consumers for doing some type of engagement. Effectively, you do an activity, you get a unit of engagement. These activities can be attention-based. Come to my talk, download my app, fill out this survey. Or capital-based. Deposit TVL into my protocol, trade on my exchange, use my credit card. Points are one specific implementation of micro work, so lightweight worker engagements. And what's important to note is points are not a phenomenon that only exists in crypto. We have points in the real world. People love them. We have free Starbucks drinks, airline miles, all kinds of ways to earn points all throughout the world. And they're fun. People like to earn points. Projects benefit from giving out points. I believe they are cool and good. In order to explain why I think points are so important to crypto today, I think it's important to look at the history of micro work in the crypto space. Back in the good old days, 2012, you could put these signatures on your Bitcoin talk profile and you would get paid Bitcoin every single time you did a forum post. You would do little things and get Bitcoin. This went like a bajillion X when ICOs launched because now you could mint free tokens that maybe would go up a lot, give those out for good work done. Of course, in 2020, things went crazy. We got the Uniswap airdrop. We got a whole bunch of airdrops.",
        "Talk Title: Actually, Points Are Cool and Good | Zane Huffman - Demos\nSpeaker: Zane Huffman\nText: But in the scope of how new malicious projects can screw you over, it's not that bad. I don't think projects have more faculty to rug users if they have access to points. With that being said, points are not perfect. Here's some ways that I think we can improve the points meta, new innovations to come, so that this can continue to be an important micro work niche that does not die out like the previous versions. Number one, accountability culture. If projects are rugging users, if they're taking their points programs and axing them or sun-signing them or pretending they don't exist, do not engage with those projects. Projects, if you're offering points, do the right thing. Additionally, we still have a lot of issues around Sybil attacks and the notion that it's difficult to assert that a participant in a project is a unique person. We need to enable the ability to vet users on the blockchain as unique individuals. This enables more nuanced engagements. For example, you can offer surveys to get acclimated to a project and you can't do a survey a thousand times.",
        "Talk Title: Actually, Points Are Cool and Good | Zane Huffman - Demos\nSpeaker: Zane Huffman\nText: There we go. Another one I see. This one mostly in regards to where people are depositing and they're locking up their assets. Points are a liquid. I'm forfeiting yield to get a point that might be worth something in the future. First of all, I think it is important that you require a higher affinity to the project you're depositing in. Additionally, there are new mechanisms in the market that get around this. On Pendle, you can separate the point and the yield, and you can speculate on the underlying asset and the opportunity to earn points separately. Whales Market also allows you to trade points over the counter before the airdrop. You're totally able to trade points even if they're illiquid, and more and more of that is going to continue to happen. I also see people posting on Twitter that they don't like earning points. These are people who post on Twitter to get likes and retweets. I think they are lying to themselves. I think points as a little hit of dopamine is way better than getting a like or a retweet or a follow. I will die on that hill. This one comes up a lot too. Points are not innovative. They exist in the real world. They have for a long time. We did not invent them in crypto. This is a good thing. We know the legal clarity around points. We have the market research that people like to earn points, that it's effective for projects and businesses.",
        "Talk Title: Actually, Points Are Cool and Good | Zane Huffman - Demos\nSpeaker: Zane Huffman\nText: Our project, Demos, helps to enable this type of proof of humanity. There are others in the space that are working on this as well. I believe that we will have a much better grasp on proof of humanity going into this next cycle. Lastly. Be intelligent about how you're offering points. Be intelligent about how you're issuing points. Just because a project has points doesn't mean they're automatically a good project. If someone forks Uniswap and they put points on top of it, it doesn't mean you need to give them your attention. Projects, if you're just looking for getting likes and retweets on Twitter, that's not a good use of points. Get people to use your app. Get people to understand the project and grow out the community. Lastly, this is the most important part of the presentation. Thank you for your attention. Claim this POAP and you will get points. Everyone who claims this POAP will get points for Demos, which will potentially be good when we launch our token. Also, the projects I'm working on at Block have points programs as well and are also pre-token. Lastly, if you guys want to find me, I'm jeffthedunker on Twitter. That's my m'lady. I'm on TikTok as well if you're on that cursed app. Thank you. I have to leave this up so people can claim it, but if anyone has questions and you're able to shout, feel free. Otherwise, come talk to me down on the side. Thank you."
    ]
    
    prompt_docs = [prompt_docs[i % 3] for i in range(20)]

    dialogs: List[Dialog] = [
        [
            {
                "role": "system",
                "content": """Below are the metrics and definitions:
off topic: Superficial or unrelevant content that can not answer the given question.
somewhat relevant: Offers partial insight to partially answer the given question.
relevant: Comprehensive, insightful content suitable for answering the given question.""",
            },
            {
                "role": "user",
                "content": f"You will be given a document and you have to rate it based on its information and relevance to the question. The document is follows:\n"
                + doc,
            },
            {
                "role": "user",
                "content": (
                    f"Use the metric choices [off topic, somewhat relevant, relevant] to evaluate whether the text can answer the given question:\n"
                    f"{query_string}"
                ),
            },
            {
                "role": "user",
                "content": "Only return a choice in lower cases for the given document. Do not return any other explanations.",
            },
        ] for doc in prompt_docs
    ]
    
    time1 = time.time()
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    time2 = time.time()
    print(f"=====> Inference time: {time2 - time1} seconds")

    for dialog, result in zip(dialogs, results):
        # for msg in dialog:
        #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
