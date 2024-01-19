from playwright.sync_api import ViewportSize, sync_playwright, Page
from urllib.parse import urlparse, urljoin
from beartype import beartype


from .actions import Action, ActionTypes
from .build_tree import HTMLTree


class HTMLEnvironment:
    @beartype
    def __init__(
        self,
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.0,
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution
        self.tree = HTMLTree()

    def setup(self, start_url: str) -> None:
        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.__enter__()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo
        )
        self.context = self.browser.new_context(
            viewport=self.viewport_size,
            device_scale_factor=1,
        )
        if start_url:
            self.page = self.context.new_page()
            self.page.goto(start_url)
            self.page.wait_for_timeout(500)
            self.html_content = self.page.content()
        else:
            self.page = self.context.new_page()
            self.html_content = self.page.content()

    def _get_obs(self):
        try:
            html_content = self.tree.fetch_html_content(self.html_content)
            tab_name = self.page.title()
            dom_tree = self.tree.build_dom_tree()
            observation = f"current web tab name is \'{tab_name}\'\n" + dom_tree
        except:
            observation = ""
        return observation

    def reset(self, start_url: str) -> str:
        self.setup(start_url)
        observation = self._get_obs()
        return observation

    def get_page(self, element_id: int):
        try:
            selector = self.tree.get_selector(element_id)
        except:
            selector = ""
        return self.page, selector

    def execute_action(self, action: Action) -> str:
        '''找到可交互元素并执行相应的动作得到新的observation'''
        try:
            match action["action_type"]:
                case ActionTypes.CLICK:
                    try:
                        label, element_idx = self.tree.get_tag_name(
                            self.tree.elementNodes[action["element_id"]])
                        action.update({"element_id": element_idx,
                                       "element_name": label})
                        selector, xpath = self.tree.get_selector_and_xpath(
                            action["element_id"])
                    except Exception as e:
                        print(
                            f"selector:{selector},label:{label},element_idx: {element_idx}")
                    if label == "link":
                        try:
                            element = self.tree.elementNodes[element_idx]
                            url = element["attributes"].get("href")
                            if bool(urlparse(url).netloc) is False:
                                base_url = self.page.url()
                                url = urljoin(base_url, url)
                            self.page = self.context.new_page()
                            self.page.goto(url)
                            self.page.wait_for_load_state('load')
                            self.html_content = self.page.content()
                            return self._get_obs()
                        except:
                            try:
                                self.page.evaluate('''() => {
                                    const element = document.querySelector('%s');
                                    if (element) {
                                        element.click();
                                    }
                                }''' % selector)
                                # await self.page.locator(selector).click()
                                self.page.wait_for_load_state('load')
                                self.html_content = self.page.content()
                                return self._get_obs()
                            except Exception as e:
                                print(e)
                    else:
                        try:
                            self.page.evaluate('''() => {
                                const element = document.querySelector('%s');
                                if (element) {
                                    element.click();
                                }
                            }''' % selector)
                            # await self.page.locator(selector).click()
                            self.page.wait_for_load_state('load')
                            self.html_content = self.page.content()
                            return self._get_obs()
                        except Exception as e:
                            print(e)
                case ActionTypes.GOTO:
                    try:
                        self.page = self.context.new_page()
                        self.page.goto(action["url"])
                        self.page.wait_for_load_state('load')
                        self.html_content = self.page.content()
                        return self._get_obs()
                    except Exception as e:
                        print("can't execute goto action")
                        print(e)
                        return ""
                case ActionTypes.FILL_FORM:
                    try:
                        try:
                            label, element_idx = self.tree.get_tag_name(
                                self.tree.elementNodes[action["element_id"]])
                            action.update({"element_id": element_idx,
                                           "element_name": label})
                            selector, xpath = self.tree.get_selector_and_xpath(
                                action["element_id"])
                        except Exception as e:
                            print(
                                f"selector:{selector},label:{label},element_idx: {element_idx}")
                        try:
                            self.page.locator(selector).fill(action["fill_text"])
                            self.page.locator(selector).press("Enter")
                            self.page.wait_for_load_state('load')
                            self.html_content = self.page.content()
                            return self._get_obs()
                        except:
                            fill_and_press_enter = '''() => {
                                        const element = document.querySelector('%s');
                                        if (element) {
                                            element.value = '%s';
                                            element.dispatchEvent(new Event('input', { bubbles: true }));
                                            element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
                                        }
                                    }
                                ''' % (selector, action['fill_text'])
                            self.page.evaluate(fill_and_press_enter)
                            self.page.wait_for_load_state('load')
                            self.html_content = self.page.content()
                            return self._get_obs()
                    except Exception as e:
                        print("can't execute fill form action")
                        print(e)
                        return ""
                case ActionTypes.GOOGLE_SEARCH:
                    try:
                        self.page = self.context.new_page()
                        self.page.goto("https://www.google.com/search?q="+action["fill_text"])
                        self.page.wait_for_load_state('load')
                        self.html_content = self.page.content()
                        return self._get_obs()
                    except Exception as e:
                        print("can't execute google search action")
                        print(e)
                case _:
                    raise ValueError(
                        f"Unknown action type {action['action_type']}"
                    )
        except Exception as e:
            print("execute action error")
            print(e)
        return ""