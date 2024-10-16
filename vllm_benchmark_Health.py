import asyncio
import time
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
import logging
import argparse
import json
import random
import csv
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SHORT_PROMPTS = [
    "my husband was working on a project in the house and all of a sudden a bump about the size of a half dollar appeard on his left leg inside below the knee. He is 69 years old and had triple by pass surgery 7 years ago. It stung when it first happened. Doesn t hurt now. He is seated with his leg ellevated. Is this an emergency?",
    "Hi, I had a subarachnoid bleed and coiling of brain aneurysm last year. I am having some major bilateral temple pain along with numbness that comes and goes in my left arm/hand/fingers. I have had headaches since the aneurysm, but this is different. Also, my moods have been horrible for the past few weeks.",
    "for the pst 6 days i have been haing upper abdome distress ... i have had gall bladder surgery 2 yrs ago ..in the past 2 yrs when i get these bouts .. i go to emerg they run cardiograms xrays ,blood work ,all comes back normal ...im stressing out over this as im scared every time it happens im having a heart attack wht is it ?",
    "Im 28 year old and suffering from male baldness pattern. I started using RICHFEEL tricology centres saw palmetto treatment but it didnt help, then I tried Minoxidil 5% but discontinued it after 12 days after having some side effects (some tiny boils on my forehead). Kindly suggest me which medicine i should take.",
    "Hi. My 18month old has had red and swollen eyes for about 5 days. I took him to his doctor and they have no idea what is wrong. His eye itself is unaffected. It seems to be isolated to the outside of the eye (eye lid, and beneath. Just last night he broke out in a rash all over his body. I need ANSWERS badly. Can you help?",
    "I fell down three stairs on my buttocks, when I got up I almost passed out I ran to my bed and waited till I felt better. I have been feeling nauseated since then and cant stand for a long period of time without feeling like im going to be sick. It is also painful to sit. Im on blood thinners what should I do?",
    "Hi, i ve been on the Noriday pill for 10 months now (i was on the Depo Injection prior to it), i started having heavy periods at the start of December and i m on my fourth period now. Is this normal for this pill? The reason i went on this particular one was because it was similiar to the Depo Injection. I didnt get any periods with the Depo.",
    "pcod irregular periods 37x31mm cyst in LT ovary on11july its my first day. 7x7mm inRTovary and 11x12mm inLT ovary on22 july 12day. 11x12mm inRTovary and 8x9mm inLT ovary on27july 17day. 1 what does this mean? 2 IS ovulation happen this time? 3 slow and late growth of follicle as in my case .ISany chance of getting preganant? 4 what to do now? 5 plssssssssssss answer soon",
    "I had a wisdom tooth extracted nearly 3 weeks ago. unfortunately some of the block went into my jaw muscle and I now have trismus. More alarming though is that my facial muscles feel very much weekened and smiling is an effort. Also I have had twitching above my lip and up cheek and under eye. I am very concerned as I am a music teacher. please help",
    "Hi iI have been diagnoed with thymic cancer, I do not like to take the drugs for pain that are prescribed,as they have side effects that are harsh.So instead i have been taking ibuprophen .for abot 4 months ,as dosage recommended, I have beenexperiencing respitory issues lately and wonder if this is the cause.I have been doing radiation and chemo treaments ,it has shown improvement in all aspects except for my breathing .I read an article on cleveland clinic facebook this mourning that said this might be why,,,...Thoughts on this please? ths ccsc",
    "What is he progression of mantle cell lymphoma? My 74 year old husband had only 2 treatments when the oncologist determined the treatment was not working...his platelets were 72 and his WBC was 358,000. They basically told him to go home and die. That was a month ago. He has become steadily weaker. He has an appetite, but cannot eat much at one sitting. He has edema in his feet. He is now complaining of a pain in his upper right quadrant; somewhere around his under arm and radiates to the back of his shoulder blades.We are totally clueless.",
    "I m a type 2 diabetic. My doctor changed my meds to three pills 5 months ago, Metformin gave extreme gastro side effects. So I have been taking Januvia, Glimeperide and Glipezide. It kept my blood sugar much lower but not perfect. I had 3 episodes of symptoms of low blood sugar but I never got a reading lower than 80. Now my pharmacists has questioned the 3 medication so, as of 3 weeks ago I no longer take Glimeperide. Now I m having mild pressure in my abdomen and sternem, shallow breathing and muscle tension for no reason. Any thoughts?",
    "Hi, I m using shades like lipsticks , gloss, lips balm for more than 5 yrs,....but for past one year, m getting allergie to those products...I seriouly don t know the reason which is making my lips itch & irritate...now-a-days whenever m using lip shades, it will automatically cause my lips & lipline - dry , itch and irritates....so stop to use lipshades and switch on to use lip balm, since lips became very dry if m not applying any mositure on it ..[ Using products are Lakme & Lotus ]...Previously i don t have this kind of problem...Only recently it is causing....Can anyone assist me the cause of my problem... Thanks.",
    "I feel pain and lifeless sort of feeling in my both legs, below knee area. Mostly in calf muscles. I am having this pain for more than 3 years now and 24 hours a day. In the morning, its hard to get up and do something. I am really low in energy levels all the time. Standing for long, intensifies the pain. I have consulted many doctors, neurologists, they gave many stress relieving medicines along with vitamins but nothing helped even a bit. I want to do exercises but there is literally no energy in me due to these prolonged pain. I feel low and down through out the day at work and at home too. Please advise.",
]

LONG_PROMPT_PAIRS = [
    {
        "prompt": "my husband was working on a project in the house and all of a sudden a bump about the size of a half dollar appeard on his left leg inside below the knee. He is 69 years old and had triple by pass surgery 7 years ago. It stung when it first happened. Doesn t hurt now. He is seated with his leg ellevated. Is this an emergency?",
        "context": "Hello. It could be a blood collection due to minor injury or a vein rupture which is also common at this age. It is not an emergency, but you should apply compression bandage and warm compresses if six hours have past. Furthermore, it should get relieved over the next few days but if it continues to increase or persist then you should see a Doctor who can examine the patient. Take care. Chat Doctor."
    },
    {
        "prompt": "Hi, I had a subarachnoid bleed and coiling of brain aneurysm last year. I am having some major bilateral temple pain along with numbness that comes and goes in my left arm/hand/fingers. I have had headaches since the aneurysm, but this is different. Also, my moods have been horrible for the past few weeks.",
        "context": "Aneurysm in brain causes headache due to compression of pain sensitive structure in brain. But pain is usually unilateral not bilateral. This pain will not radiate in both upper limb. Bilateral upper limb pain may be due to cervical spondylosis or other causes. You have anxiety of aneurysm, so due to anxiety your mood is horrible because aneurysm does not cause horrible mood change. So your headache may be chronic tension type headache. You may relieve by antianxiety Chat Doctor."
    },
        {
        "prompt": "for the pst 6 days i have been haing upper abdome distress ... i have had gall bladder surgery 2 yrs ago ..in the past 2 yrs when i get these bouts .. i go to emerg they run cardiograms xrays ,blood work ,all comes back normal ...im stressing out over this as im scared every time it happens im having a heart attack wht is it ?",
        "context": "Hi. Thanks for your query, read and understood your problems. You are getting bouts of pain in the upper abdomen within the last 2 years particularly after the gall bladder surgery. This is great news that cardiac problems are ruled out by the tests every time in the emergency Room. This means that we have to find a local cause in the abdomen itself. I would suggest the following"
    },
    {
        "prompt": "Im 28 year old and suffering from male baldness pattern. I started using RICHFEEL tricology centres saw palmetto treatment but it didnt help, then I tried Minoxidil 5% but discontinued it after 12 days after having some side effects (some tiny boils on my forehead). Kindly suggest me which medicine i should take.",
        "context": "Hi, You may stop it for a few days. Take oral treatment like tab finasteride 1 mg daily with tab biotin 10 mg thrice a day. Cleanse the scalp with ketoconazole shampoo daily. After a few days you may restart the minoxidil. Tiny boils may be folliculitis. Take a course of antibiotic like cap Doxycycline. Avoid stress and worries. Continue treatment for long time to get good result in an Chat Doctor. Hope I have answered your query. Let me know if I can assist you further."
    },
    {
        "prompt": "Hi. My 18month old has had red and swollen eyes for about 5 days. I took him to his doctor and they have no idea what is wrong. His eye itself is unaffected. It seems to be isolated to the outside of the eye (eye lid, and beneath. Just last night he broke out in a rash all over his body. I need ANSWERS badly. Can you help?",
        "context": "Hello. I'm a pediatrician. From your description I think he might be having present orbital cellulitis. In this condition soft tissue around eye is infected. Baby will be toxic and having high fever. Eyes may be swollen, it massive be difficult to open eye. Rash on rest body might be due to Chat Doctor. Consult ophthalmologist and pediatrician. If it is really orbital cellulitis then baby might be needing admission"
    },
    {
        "prompt": "I fell down three stairs on my buttocks, when I got up I almost passed out I ran to my bed and waited till I felt better. I have been feeling nauseated since then and cant stand for a long period of time without feeling like im going to be sick. It is also painful to sit. Im on blood thinners what should I do?",
        "context": "Dear France pain is from muscular strain due to injury. You can take analgesic like ibuprofen or if severe pain take combination of diclofenac+paracetamol+chlorzoxazon/serratiopeptidase with omeprazole/pantoprazole + risperidone to relieve nausea for 2-3 days. This will clear all symptoms. If symptoms are severe consult your family doctor first. Hope this answer will be helpful to you. For more queries please don't hesitate to ask. Get well soon."
    },
    {
        "prompt": "Hi, i ve been on the Noriday pill for 10 months now (i was on the Depo Injection prior to it), i started having heavy periods at the start of December and i m on my fourth period now. Is this normal for this pill? The reason i went on this particular one was because it was similiar to the Depo Injection. I didnt get any periods with the Depo.",
        "context": "Hi, Thanks for the query. Holiday pill is a mini pill containing progesterone. During the usage of contraceptive pills some amount of irregularity in menstrual flow is common. If you are having heavy bleeding, you can take symptomatic treatment for that and can continue the pills. If same problem repeats frequently better to think of changing the contraceptive method. You discuss this with your doctor and proceed according to her advice. Take care."
    },
    {
        "prompt": "pcod irregular periods 37x31mm cyst in LT ovary on11july its my first day. 7x7mm inRTovary and 11x12mm inLT ovary on22 july 12day. 11x12mm inRTovary and 8x9mm inLT ovary on27july 17day. 1 what does this mean? 2 IS ovulation happen this time? 3 slow and late growth of follicle as in my case .ISany chance of getting preganant? 4 what to do now? 5 plssssssssssss answer soon",
        "context": "Hi welcome to Chat Doctor hi Joshi., the above report stating the measurement of the largest cyst is 37by31 in left ovary... .the other follicles are of different sizes respect to the duration. Even though many follicles develop, only one may ovulate or many at a time hence fertilization will be disturbed... you can have coitus with your husband.... and then after a week you need to follow up by USG. Hope I have answered your question. Take care."
    },
    {
        "prompt": "I had a wisdom tooth extracted nearly 3 weeks ago. unfortunately some of the block went into my jaw muscle and I now have trismus. More alarming though is that my facial muscles feel very much weekened and smiling is an effort. Also I have had twitching above my lip and up cheek and under eye. I am very concerned as I am a music teacher. please help",
        "context": "Thanks for your query, I have gone through your query. The twitching over the facial muscles and Erasmus can be because of the nerve injury to facial nerve while injecting the local anesthesia. The Erasmus is because of the injury to the muscles of mastication while injecting the local anesthesia. Nothing to worry, the twitching will come down gradually over a week. You can take nerve regenerating Chat Doctor. For the Erasmus, you can take a muscle relaxant like chlorzoxazone. I hope my answer will help you, take care."
    },
    {
        "prompt": "Hi iI have been diagnoed with thymic cancer, I do not like to take the drugs for pain that are prescribed,as they have side effects that are harsh.So instead i have been taking ibuprophen .for abot 4 months ,as dosage recommended, I have beenexperiencing respitory issues lately and wonder if this is the cause.I have been doing radiation and chemo treaments ,it has shown improvement in all aspects except for my breathing .I read an article on cleveland clinic facebook this mourning that said this might be why,,,...Thoughts on this please? ths ccsc",
        "context": "Thy mic cancer causes respiratory distress. After a thy mic tumor is found and tests have been done to get a sense of its Factors important in choosing a treatment include the type and stage of the cancer, whether it is respectable (able to be completely removed with surgery), and whether you have any other serious medical problems. Because thy mic cancer is rare, it is often hard to accurately predict the effectiveness of treatment strategies, and in many cases the best way to treat this cancer is still not preselecting a treatment plan is an important decision, and you should take the time to think about all of your choices. The main treatments for thymus cancer are"
    },
    {
        "prompt": "What is he progression of mantle cell lymphoma? My 74 year old husband had only 2 treatments when the oncologist determined the treatment was not working...his platelets were 72 and his WBC was 358,000. They basically told him to go home and die. That was a month ago. He has become steadily weaker. He has an appetite, but cannot eat much at one sitting. He has edema in his feet. He is now complaining of a pain in his upper right quadrant; somewhere around his under arm and radiates to the back of his shoulder blades.We are totally clueless.",
        "context": "Thanks for your question on Chat Doctor. Mentle cell lymphoma is one of the most aggressive tumor known till day. It is resistant to chemotherapy and rapidly worsening in days only. In your husbands' case as he is having right sided back and chest pain, I think he is developing malignant effusion. So better to rule out this by chest x-ray. If effusion than pigtail catheter insertion will help him symptomatically. So get done chest x-ray and pigtail Chat Doctor. Also discuss end of life issues with him. This will give him more support."
    },
    {
        "prompt": "I m a type 2 diabetic. My doctor changed my meds to three pills 5 months ago, Metformin gave extreme gastro side effects. So I have been taking Januvia, Glimeperide and Glipezide. It kept my blood sugar much lower but not perfect. I had 3 episodes of symptoms of low blood sugar but I never got a reading lower than 80. Now my pharmacists has questioned the 3 medication so, as of 3 weeks ago I no longer take Glimeperide. Now I m having mild pressure in my abdomen and sternem, shallow breathing and muscle tension for no reason. Any thoughts?",
        "context": "Hi sir. Usually diabetic medicines with metformin may cause abdominal problems like gastritis, ulcer.make sure that your sugar level is under control especially HE a1c is under control. Pain in lower abdomen and sternum is exactly due to gastritis so it's better to take anti ulcer medicines to solve this problem avoid taking oily and spicy foods avoid taking tea coffee or milk avoid some vegetables like radish cabbage and cauliflower so that it may help to reduce gastritis. Consult a physician for medications, thank you."
    },
    {
        "prompt": "Hi, I m using shades like lipsticks , gloss, lips balm for more than 5 yrs,....but for past one year, m getting allergie to those products...I seriouly don t know the reason which is making my lips itch & irritate...now-a-days whenever m using lip shades, it will automatically cause my lips & lipline - dry , itch and irritates....so stop to use lipshades and switch on to use lip balm, since lips became very dry if m not applying any mositure on it ..[ Using products are Lakme & Lotus ]...Previously i don t have this kind of problem...Only recently it is causing....Can anyone assist me the cause of my problem... Thanks.",
        "context": "Hi, welcome to the Chat Doctor forum, the complaints which you mentioned is called as contact dermatitis or chemical irritant dermatitis, it has an allergic component, since you are getting this for the past 1 year it signifies that your immunity which prevents from getting this kind of allergies is getting down to lower levels, I suspect you are under some kind of tremendous stress either financial or mental which is causing this type of hypersensitivity in you, this is absolutely curable with special homeopatChatDoctoredicines.... get well soon ! You can contact me on Chat Doctor. Com"
    },
    {
        "prompt": "I feel pain and lifeless sort of feeling in my both legs, below knee area. Mostly in calf muscles. I am having this pain for more than 3 years now and 24 hours a day. In the morning, its hard to get up and do something. I am really low in energy levels all the time. Standing for long, intensifies the pain. I have consulted many doctors, neurologists, they gave many stress relieving medicines along with vitamins but nothing helped even a bit. I want to do exercises but there is literally no energy in me due to these prolonged pain. I feel low and down through out the day at work and at home too. Please advise.",
        "context": "Dear patient you need detailed blood check up for complete blood count with ESR level. Low hemoglobin level may be the reason for lethargy and weakness. Pain in the Lower limbs may be due to spine pathology with nerve compression. First X-ray of the lumbosacral spine anteroposterior and lateral views and them MRI of the lumbosacral spine with screening of whole spine needs to be done. Please get it done from radiology center nearby you. Meanwhile, start tab regain x 75 mg one at bedtime with tab attract twice a day for pain relief. If report is abnormal you need to consult orthopedic surgeon nearby your area with report."
    },
    {
        "prompt": "19 y/o female 110 lbsI, no significant medical bg. was hit in the chest with a blunt object right above my left breast about 1 week ago. Now when I breathe in too deeply, sneeze, cough, or move my left arm in a certain way, I feel discomfort/pain. The pain comes from sneezing. Also now there is quarter-size bump and the pain travels to my armpit/side area at certain times. When I press down on the bump I feel some pain. It is a hard bump but no bruising. When my chest got hit I felt winded. Thank you for any answers",
        "context": "Thanks for your question on Chat Doctor. I can understand your concern. You had blunt chest trauma and symptoms you are having at present are due to musculoskeletal injury after such trauma. So follow these steps for better symptomatic relief in musculoskeletal pain. 1. Avoid movements causing pain. 2. Avoid bad postures in sleep. 3. Avoid heavyweight lifting and strenuous exercise. 4. Take painkiller and anti-inflammatory Chat Doctor. 5. Apply warm water pad on affected areas. Don't worry, swelling and pain will subside with all these in 1-2 weeks. Hope I have solved your query. I will be happy to help you further. Wish you good health. Thanks."
    },
    {
        "prompt": "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.. Thank you doc!",
        "context": "Hi, Thank you for posting your query. The most likely cause for your symptoms is benign paroxysmal positional vertigo (BPPV), a type of peripheral vertigo. In this condition, the most common symptom is dizziness or giddiness, which is made worse with movements. Accompanying nausea and vomiting are common. The condition is due to problem in the ear, and improves in a few days on own. Betahistine tablets would help relieve your symptoms. Doing vestibular rehabilitation or adaptation exercises would prevent the recurrence of these symptoms. An ENT evaluation would also help. I hope it helps. Best wishes, Chat Doctor."
    },
    {
        "prompt": "My grandmother who is 87 had an injury to her nose as result of falling out of her wheelchair. She did lose alot of blood due to blood thinner. This accident ocurred 2 weeks ago. She is doing better but her H&H levels are low 8.2 was 8.7 2 days ago and 25.2% They want to send her for blood transfusions but I have read risks and not confident this is the way to go. They are taking her blood daily and they took it this morning hence the lower levels and they started her back on coumedin therapy too. Can her H&H levels be lower in the am?",
        "context": "Hi, Thanks for asking. Based on your query, my opinion is as follows.1. Both reduced hemoglobin and reduced hematocrit indicates anemia.2. However, 8.2g% is not significant, unless she has associated co-morbidities like cardiac failure.3. If she has cardiac problems or any breathing difficulty, she will require packed red cell transfusion to avoid further complications. Daily blood Chat Doctor. Continue all medications. She needs to be on blood thinner to avoid heart attack or stroke. Hope it helps. Any further queries, happy to help again."
    },
    {
        "prompt": "I also have been experiencing a lower left abdominal vibration, no pain. Started on July 18th. I had been painting and scraping the garage, on a ladder. thought maybe this triggered something. It went on, off and on for 2 weeks. We went to visit our son and NYC. Put on 45 miles walking. no problem. Came home. Off and on vibration started again, then stopped completely for ;7 days. Now back again. I am 64, weight about 148, take advair for asthma. The vibration is annoying and scary because I don t know what is causing it and hoped after it had stopped for 7 days that it would go away on its own.",
        "context": "Hi. Regards. Thanks for an elucidated history, which say you have walked 45 miles too. This indicates that there is no intra-abdominal problem. The vibration in the left lower abdomen looks to be due to the tics of the musculature of that part of the abdominal wall possibly due to a pinched nerve, or an automatic discharge of the nerve or may be a disc in the thoracic region. Get examined by an Orthopedic Surgeon / General Surgeon to get a physical examination done, an ultrasonography of the abdomen and an MRI of the thoracolumbar spine. Relax, this is not a serious problem, although it is definitely an irritating one."
    },
    {
        "prompt": "Hi Dr. P.Koregol, I am Dick Lim age 67. Thank you for looking into my query. These result dated Apr 14; bracket was Jun 2013. My Cholesterol- LDL 64 (144). High HDL 156 (68) mg/dl n yet Total Cholesterol 240 (227) mg/dl. T 99 (75) mg/dl, Total Chol/HDL-Chol 1.5 (3.3); Ph 8.0 (7.5). Blood Monocytes 7% (6&); Lymphocytes 880 (882)/cmm. BP- 140/85 (below). Blood sugar fasting- 100 (86) Free T 0.96 (0.99) ng/dl. Bilirubin 1.7 (2.1) v mg/dl Exercise 2 to 4 times a week about an hr. each time. Kindly give whatever advises you can offer will be much appreciated.",
        "context": "Hello, All your blood investigation are in normal range except the lipids. Now for you, we will want total cholesterol under 200 and LDL cholesterol under 130 at least and preferably under 100. The good cholesterol HDL is normal and we shouldn't worry about the same. Low how would we do it? Atleast 45 mins of aerobic exercise Daily Cut on oils, red meat, egg yellow, fried and preserved food. May shift to healthy oils like olive, 2 serving of fish /per week. Supplement like omega 3 fatty acid 1-2 GMS a day, flax seed may work No medication required for any of your reports including lipids. After all lifestyle changes repeat a lipid profile after 3 months. Regards"
    },
]

async def process_stream(stream):
    stream_message = ''
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
            stream_message += chunk.choices[0].delta.content
        if chunk.choices[0].finish_reason is not None:
            break
    logging.info(stream_message)
    logging.info(f'total tokens={total_tokens}')
    return first_token_time, total_tokens

async def make_request(client, output_tokens, request_timeout, use_long_context):
    start_time = time.time()
    if use_long_context:
        prompt_pair = random.choice(LONG_PROMPT_PAIRS)
        content = prompt_pair["context"] + "\n\n" + prompt_pair["prompt"]
    else:
        content = random.choice(SHORT_PROMPTS)
    logging.info(content)
    try:
        stream = await client.chat.completions.create(
            model="/LLMConf/BaseLLM/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": content}],
            max_tokens=output_tokens,
            stream=True
        )
        
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        return total_tokens, elapsed_time, tokens_per_second, ttft

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None

async def worker(client, semaphore, queue, results, output_tokens, request_timeout, use_long_context):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break
            logging.info(f"Starting request {task_id}")
            result = await make_request(client, output_tokens, request_timeout, use_long_context)
            if result:
                results.append(result)
            else:
                logging.warning(f"Request {task_id} failed")
            queue.task_done()
            logging.info(f"Finished request {task_id}")

def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)

async def run_benchmark(num_requests, concurrency, request_timeout, output_tokens, vllm_url, api_key, use_long_context):
    client = AsyncOpenAI(base_url=vllm_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []

    # add tasks to the queue
    for i in range(num_requests):
        await queue.put(i)
    
    # add termination signals to stop worker threads
    for _ in range(concurrency):
        await queue.put(None)

    # create worker threads
    workers = [asyncio.create_task(worker(client, semaphore, queue, results, output_tokens, request_timeout, use_long_context)) for _ in range(concurrency)]

    start_time = time.time()
    
    # wait for all tasks to complete
    await queue.join()
    await asyncio.gather(*workers)

    end_time = time.time()

    # calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _ in results if tokens is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft in results if ttft is not None]
    tpot_list = [x / y for x, y in zip(latencies, [tokens for tokens, _, _, _ in results if tokens is not None])]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_tpot = sum(tpot_list) / len(tpot_list) if tpot_list else 0

    # calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]
    tpot_percentiles = [calculate_percentile(tpot_list, p) for p in percentiles]

    results_dict = {
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "use_long_context": use_long_context,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token(ttft)": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        },
        "time-per-output-token(tpot)": {
            "average": avg_tpot,
            "p50": tpot_percentiles[0],
            "p95": tpot_percentiles[1],
            "p99": tpot_percentiles[2]
        }
    }

    # write results to CSV file
    csv_file = '/LLMConf/data/data.csv'
    file_exists = os.path.isfile(csv_file)
    
    # read existing data
    if file_exists:
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame()

    # create new row data
    new_row = {
        "latency_average": avg_latency,
        "latency_p50": latency_percentiles[0],
        "latency_p95": latency_percentiles[1],
        "latency_p99": latency_percentiles[2],
        "tokens_per_second_average": avg_tokens_per_second,
        "tokens_per_second_p50": tps_percentiles[0],
        "tokens_per_second_p95": tps_percentiles[1],
        "tokens_per_second_p99": tps_percentiles[2],
        "time_to_first_token_average": avg_ttft,
        "time_to_first_token_p50": ttft_percentiles[0],
        "time_to_first_token_p95": ttft_percentiles[1],
        "time_to_first_token_p99": ttft_percentiles[2],
        "time_per_output_token_average": avg_tpot,
        "time_per_output_token_p50": tpot_percentiles[0],
        "time_per_output_token_p95": tpot_percentiles[1],
        "time_per_output_token_p99": tpot_percentiles[2]
    }

    # add new row data to the existing data
    if not df.empty:
        # ensure that the existing data has at least 35 columns
        for column in new_row.keys():
            if column not in df.columns:
                df[column] = pd.NA

        # update the existing rows from column 11 to column 35.
        last_index = len(df) - 1
        for column, value in new_row.items():
            df.at[last_index, column] = value
    else:
        # update the existing rows from column 11 to column 35
        df = pd.DataFrame([new_row])

    # save the updated data to the CSV file
    df.to_csv(csv_file, index=False)

    return results_dict

def print_results(results):
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLaMA-3 model with vLLM")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to make")
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent requests")
    parser.add_argument("--request_timeout", type=int, default=30, help="Timeout for each request in seconds (default: 30)")
    parser.add_argument("--output_tokens", type=int, default=50, help="Number of output tokens (default: 50)")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(args.num_requests, args.concurrency, args.request_timeout, args.output_tokens, args.vllm_url, args.api_key, args.use_long_context))
    print_results(results)
else:
    # When imported as a module, provide the run_benchmark function
    __all__ = ['run_benchmark']

