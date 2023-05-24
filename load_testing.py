import random
from locust import HttpUser, task, between


class WebUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def deployment_page(self):
        self.client.get(url="/")

    @task
    def deployment_page(self):
        random_texts = [
            "Lorem ipsum dolor sit amet.",
            "Consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        ]
        user_text = random.choice(random_texts)
        # json_input = json.dumps({'sentence': random.choice(user_text)})
        self.client.post(url="/predict", data={"text": user_text})
