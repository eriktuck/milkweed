from dotenv import load_dotenv
import os
import asyncio
from pathlib import Path
from monarchmoney import MonarchMoney

env_path = Path('./secrets/.env')
load_dotenv(dotenv_path=env_path)

async def main():
    mm = MonarchMoney()
    mm._headers['Device-UUID'] = '98d6a448-4798-437f-9927-950f643da374'

    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")

    await mm.login(email=username, 
                   password=password, 
                   use_saved_session=False, 
                   save_session=False)
    await mm.get_transactions(start_date="2024-01-01", end_date="2024-12-31")

asyncio.run(main())