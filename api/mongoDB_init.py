import pymongo
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus
load_dotenv()

mongodb_password = os.environ['mongodb_password']
mongodb_password = quote_plus(mongodb_password)
connect_string = f'mongodb+srv://hdttuan:{mongodb_password}@cluster0.h09da4d.mongodb.net/?retryWrites=true&w=majority'
crawlClient = pymongo.MongoClient(connect_string)
crawlClient = crawlClient['TRACKINGINVESTMENT_CRAWL']


mongodb_main2_password = os.environ['mongodb_main2_password']
mongodb_main2_password = quote_plus(mongodb_main2_password)
connect_string = f'mongodb+srv://thanhtuanhuynh0011:{mongodb_main2_password}@cluster0.vshugfj.mongodb.net/TRACKSCAN?retryWrites=true&w=majority'
main2Client = pymongo.MongoClient(connect_string)
main2Client = main2Client['TRACKSCAN']
