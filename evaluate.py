import time
import json5
import requests
from agent.Environment.html_env.async_env import AsyncHTMLEnvironment
from evaluate import *
from agent.Plan import *
from playwright.async_api import Playwright, async_playwright, expect, Page
from agent.Environment.html_env.actions import create_action, Action, ActionTypes

import re
import asyncio
import argparse
import toml


# 解析命令行参数
parser = argparse.ArgumentParser(
    description="Run the agent in different modes.")
parser.add_argument("--mode", choices=["dom", "vision", "d_v"], default="d_v",
                    help="Choose interaction mode: 'dom' for DOM-based interaction, 'vision' for vision-based interaction, 'd_v' for DOM-based and vision-based interaction.")
parser.add_argument("--index", "--i", type=str, default=-1)
args = parser.parse_args()
interaction_mode = args.mode
raw_data_index = args.index


def read_file(path="./data/test.json"):
    '''读取标签数据'''
    return_list = []
    with open(path) as f:
        test_data = json5.load(f)
    for task in test_data:
        task_name = task["task"]
        evaluation_data = task["evaluation"]
        reference_task_length = task["reference_task_length"]
        reference_evaluate_steps = []
        for _, evaluation in enumerate(evaluation_data):
            match_function = evaluation["match_function_name"]
            if "url" in match_function:
                key = evaluation["content"]["key"]
                reference_answer = evaluation["content"]["reference_answer"]
                reference_evaluate_steps.append({"match_function": match_function,
                                                "key": key, "reference_answer": reference_answer, "score": 0})
            elif "element_path" in match_function:  # TODO
                reference_answer = evaluation["content"]["reference_answer"]
                method = evaluation["method"]
                netloc = evaluation["content"]["netloc"]
                reference_evaluate_steps.append({"match_function": match_function, "method": method,
                                                "reference_answer": reference_answer, "netloc": netloc, "score": 0})
            elif "element_value" in match_function:
                reference_answer = evaluation["content"]["reference_answer"]
                netloc = evaluation["content"]["netloc"]
                reference_evaluate_steps.append({"match_function": match_function,
                                                "reference_answer": reference_answer, "netloc": netloc, "score": 0})
        return_list.append(
            [task_name, reference_task_length, reference_evaluate_steps])
    # print(return_list)
    # return_list=return_list[1:]
    return return_list


def get_netloc(url: str) -> str:
    '''提取出域名，如zhihu.com提取出zhihu，www.google.com.hk提取出google'''
    url = urlparse(url)
    try:
        if url.netloc.startswith("www"):
            netloc = re.findall(".*?\.(.*?)\..*?", url.netloc)[0]
        else:
            netloc = re.findall("(.*?)\..*?", url.netloc)[0]
    except:
        netloc = ""
    return netloc


async def get_element_content(page: Page, selector):
    '''获取元素内容'''
    try:
        # 获取元素的标签名
        tag_name = await page.eval_on_selector(selector, "element => element.tagName.toLowerCase()")

        if tag_name in ["input", "textarea"]:
            # 对于 input 或 textarea 元素
            return await page.input_value(selector)
        else:
            # 对于其他元素
            return await page.text_content(selector)
    except:
        return ""


async def step_evaluate(page: Page, evaluate_steps=[], input_path=None):
    '''评测步骤打分'''
    # reference_evaluate_steps, num_steps
    # num_steps += 1
    # page_url = html_env.page.url
    # page_url = page.url
    step_score = 0
    for evaluate in evaluate_steps:
        if evaluate["score"] != 1:
            match_function = evaluate["match_function"]
            if match_function == "url_exactly_match":
                score = URLEvaluator.url_exact_match(
                    page.url, evaluate["reference_answer"], evaluate["key"])
            elif match_function == "url_included_match":
                score = URLEvaluator.url_include_match(
                    page.url, evaluate["reference_answer"], evaluate["key"])
            elif match_function == "url_semantic_match":
                score = await URLEvaluator.url_semantic_match(
                    page.url, evaluate["reference_answer"], evaluate["key"])
                print(score, "url_semantic_match")
            elif match_function == "element_path_exactly_match":
                input_netloc = get_netloc(page.url)
                method = evaluate["method"]
                score = ElementEvaluator.path_exact_match(
                    input_path, evaluate["reference_answer"], method, await page.content(), input_netloc, evaluate["netloc"])
                print(score, "path_exact_match:", input_path,
                      "***", evaluate["reference_answer"])
            elif match_function == "element_path_included_match":
                pass
                # * 暂时不做
                # method = evaluate["method"]
                # score = ElementEvaluator.path_included_match(
                #     input_path, evaluate["reference_answer"], method, await page.content())
            elif match_function == "element_value_exactly_match":
                if input_path is not None:
                    input_netloc = get_netloc(page.url)
                    # page_content = await page.content()
                    # soup = BeautifulSoup(page_content, 'html.parser')
                    # element_value = soup.select_one(input_path)#.text.strip()
                    # element_value = await page.input_value(input_path)
                    element_value = await get_element_content(page, input_path)
                    print(element_value)
                    # print(await page.locator(input_path).input_value())
                    score = ElementEvaluator.element_value_exact_match(
                        element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"])
                    print(score, "element_value_exactly_match",
                          element_value, "*", evaluate["reference_answer"])
            elif match_function == "element_value_included_match":
                if input_path is not None:
                    input_netloc = get_netloc(page.url)
                    # page_content = await page.content()
                    # soup = BeautifulSoup(page_content, 'html.parser')
                    # element_value = soup.select_one(input_path).text.strip()
                    # element_value = await page.input_value(input_path)
                    element_value = await get_element_content(page, input_path)
                    score = ElementEvaluator.element_value_include_match(
                        element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"])
                    print(score, "element_value_included_match",
                          element_value, "*", evaluate["reference_answer"])

            elif match_function == "element_value_semantic_match":
                if input_path is not None:
                    input_netloc = get_netloc(page.url)
                    # page_content = await page.content()
                    # soup = BeautifulSoup(page_content, 'html.parser')
                    # element_value = soup.select_one(input_path).text.strip()
                    # element_value = await page.input_value(input_path)
                    element_value = await get_element_content(page, input_path)
                    if len(element_value) > 0:
                        score = await ElementEvaluator.element_value_semantic_match(
                            element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"])
                        print(score, "element_value_semantic_match",
                              element_value, "*", evaluate["reference_answer"])
            elif match_function == "text_exact_match":
                pass  # TODO
            elif match_function == "text_include_match":
                pass
            elif match_function == "text_semantic_match":
                pass

            evaluate["score"] = max(evaluate["score"], score)
        step_score += evaluate["score"]
    print("current step score:", step_score, "/", len(evaluate_steps))
    return evaluate_steps
    # print(evaluate_steps)


async def aexec_playwright(code, page):
    '''async执行playwright代码'''
    exec(
        f'async def __ex(page): ' +
        ''.join(f'\n {l}' for l in code.split('\n'))
    )
    # Get `__ex` from local variables, call it and return the result
    return await locals()['__ex'](page)


async def main(num_steps=0, mode="dom"):

    file = read_file()

    with open('./configs/dom.toml', 'r') as f:
        config = toml.load(f)

    # 评测输入范围内的任务
    if raw_data_index != -1:
        re_result = re.split(r'\s|,', raw_data_index)
        raw_data_start_index = int(re_result[0])
        raw_data_end_index = int(re_result[-1]) + 1
    else:
        raw_data_start_index = 0
        raw_data_end_index = len(file)
    print(raw_data_start_index, raw_data_end_index)

    # for task_index in range(raw_data_start_index, raw_data_end_index):

    start_index = 22
    for task_index in range(start_index, len(file)):
        task = file[task_index]
        task_name, reference_task_length, reference_evaluate_steps = task
        print("task index:", task_index)
        print("task_name:", task_name)
        print("reference_task_length:", reference_task_length)
        print("raw data:\n", reference_evaluate_steps)
        # #! # 1. playwright
        # # 用playwright运行浏览器

        # async def run(playwright: Playwright) -> None:
        #     '''用playwright运行浏览器'''
        #     evaluate_steps = reference_evaluate_steps
        #     browser = await playwright.chromium.launch(headless=False)
        #     context = await browser.new_context(locale="en-US")
        #     page = await context.new_page()
        #     replay_codes = open("./data/playwright/google.txt", "r", encoding="utf-8")
        #     for num_steps, line in enumerate(replay_codes):
        #         print("step:", num_steps, line)
        #         selector = None
        #         if "page.locator" in line:
        #             selector = re.findall('page.locator\("(.*?)"\).*?\(.*?\)', line)[0]
        #             print("selector:", selector)
        #         line = "await "+line
        #         print(line)
        #         evaluate_steps = await step_evaluate(page=page, evaluate_steps=evaluate_steps, input_path=selector)
        #         time.sleep(3)
        #         await aexec_playwright(line, page)
        #         time.sleep(2)
        #     return num_steps, evaluate_steps

        # async with async_playwright() as playwright:
        #     num_steps, evaluate_steps = await run(playwright)

        # ! # 2. planning
        env = AsyncHTMLEnvironment(
            mode=mode,
            max_page_length=8192,
            headless=False,
            slow_mo=1000,
            current_viewport_only=False,
            viewport_size={"width": 1920, "height": 1280} if mode == "dom" else {
                "width": 1080, "height": 720},
            # "width": 1080, "height": 720
            save_trace_enabled=False,
            sleep_after_execution=0.0,
            locale="en-US",
            use_vimium_effect=True
        )

        DF = config['basic']['default']
        GR = config['basic']['global_reward']
        CR = config['basic']['current_step_reward']
        PT = config['basic']['previous_trace']

        observation_VforD = None
        if mode == "d_v":
            observation, observation_VforD = await env.reset("about:blank")
        else:
            observation = await env.reset("about:blank")

        previous_trace = []
        evaluate_steps = reference_evaluate_steps

        # task_name = "Add a blue iPad to your cart and select the option for free engraving with \"hello world\" with no other accessaries."
        last_action_description = ""
        dict_to_write = None
        for action_step in range(config['basic']['Max_Action_Step']):
            total_step_score = 0
            # break
            print("planning前previous_trace：", previous_trace)
            print("planning前observation：", observation)
            for _ in range(3):
                try:
                    if DF:
                        dict_to_write = await Planning.plan(uuid=1, user_request=task_name, previous_trace=previous_trace, observation=observation, feedback=last_action_description, mode=mode, observation_VforD=observation_VforD)
                        if dict_to_write is not None:
                            break
                    elif GR == False:
                        dict_to_write = await Planning.plan(uuid=1, user_request=task_name, previous_trace=previous_trace, observation=observation, feedback=last_action_description, mode=mode, observation_VforD=observation_VforD, global_reward=False)
                        if dict_to_write is not None:
                            break
                    elif CR == False:
                        dict_to_write = await Planning.plan(uuid=1, user_request=task_name, previous_trace=previous_trace, observation=observation, feedback="", mode=mode, observation_VforD=observation_VforD)
                        if dict_to_write is not None:
                            break
                    elif PT == False:
                        dict_to_write = await Planning.plan(uuid=1, user_request=task_name, observation=observation, feedback=last_action_description, mode=mode, observation_VforD=observation_VforD)
                        if dict_to_write is not None:
                            break
                except Exception as e:
                    traceback.print_exc()
                    continue

            def parse_current_trace(response):
                thought = response["description"].get("thought")
                action_type = response['action_type']
                acton_input = response['value']
                action = response["description"].get("action")
                current_trace = {"thought": thought, "action": action}
                try:
                    element_id = int(response['id'])
                except:
                    element_id = 0
                #! env.tree.nodeDict[element_id]勿动，调用映射关系，否则selector会出错
                if action_type in ["fill_form", "fill_search", "click"]:
                    try:
                        selector = env.tree.get_selector_and_xpath(
                            env.tree.nodeDict[element_id])
                    except:
                        print("accessibility tree don't have this element_id")
                        selector = None
                        element_id = 0
                        action_type = "None"
                else:
                    selector = None
                    element_id = 0
                execute_action = create_action(
                    elementid=element_id, action_type=action_type, action_input=acton_input)
                return execute_action, current_trace, selector
            print("dict_to_write:", dict_to_write)

            if mode == "dom" or mode == "d_v":
                execute_action, current_trace, path = parse_current_trace(
                    dict_to_write)
                selector, xpath = (
                    path[0], path[1]) if path is not None else (None, None)
                print("current trace:\n", current_trace)
                print("response:\n", execute_action)
                print("selector:", selector)

                evaluate_steps = await step_evaluate(page=env.page, evaluate_steps=evaluate_steps, input_path=selector)
                print("执行动作前的url", env.page.url)
                for evaluate in evaluate_steps:
                    total_step_score += evaluate["score"]
                print(total_step_score, "/", len(reference_evaluate_steps))
                if total_step_score == len(reference_evaluate_steps):
                    break
                # input()
                if mode == "d_v":
                    observation, observation_VforD = await env.execute_action(execute_action)
                else:
                    observation = await env.execute_action(execute_action)
                print("执行动作后的url", env.page.url)

            elif mode == "vision":
                execute_action = dict_to_write["action"]
                thought = dict_to_write["description"].get("thought")
                action = dict_to_write["description"].get("action")
                current_trace = {"thought": thought, "action": action}
                print("执行动作前的url", env.page.url)
                if await env.vision_execute_action(execute_action):
                    break
                print("vision_execute_action finished!")
                observation = await env._get_obs()
                print("执行动作后的url", env.page.url)

            if mode == "dom" or mode == "d_v":
                # current_trace = [current_trace]
                current_reward = await Planning.evaluate(user_request=task_name, previous_trace=previous_trace,
                                                         current_trace=current_trace, observation=observation)
                if current_reward and int(current_reward.get("score")) < config['basic']['Step_Score_Threshold']:
                    execute_action.update(
                        {"element_id": 0, "action_type": ActionTypes.GO_BACK})
                    if mode == "d_v":
                        observation, observation_VforD = await env.execute_action(execute_action)
                    else:
                        observation = await env.execute_action(execute_action)
                    last_action_description = current_reward.get("description")
                else:
                    last_action_description = ""
                    previous_trace.append(current_trace)
            elif mode == "vision":
                previous_trace.append(current_trace)
                if dict_to_write["description"].get('reward'):
                    if "loop" in dict_to_write["description"].get('reward').get("status"):
                        previous_trace = []
                        previous_trace.append(current_trace)

            a = input("回车继续下一个Action，按q退出")
            if a == "q":
                break
        # a = await Planning.plan(uuid=1, user_request="Find Dota 2 game and add all DLC to cart in steam.")
        # print(json5.dumps(a, indent=4))
        # input()

        # ! 3.任务评测打分
        if mode == "dom" or mode == "d_v":
            # step score
            total_step_score = 0
            for evaluate in evaluate_steps:
                total_step_score += evaluate["score"]
            print("\ntotal step score:", total_step_score,
                  "/", len(reference_evaluate_steps))

            # length score
            task_evaluator = TaskLengthEvaluator()
            task_length_score = task_evaluator.task_length_score(
                reference_task_length, num_steps)
            print("task_length_score:", task_length_score)

            # finish score
            finish_task_score = FinishTaskEvaluator.finish_task_score(
                len(reference_evaluate_steps), total_step_score)
            print("finish_task_score:", finish_task_score)

        a = input("回车继续，按q退出")
        await env.close()
        del env
        if a == "q":
            break

    print(f"\033[31mtask finished!\033[0m")  # 红色
    input(f"\033[31m按回车键结束\033[0m")


if __name__ == "__main__":
    asyncio.run(main(mode=interaction_mode))
