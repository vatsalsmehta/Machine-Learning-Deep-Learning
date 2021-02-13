import scrapy
import xlsxwriter
import scrapy.crawler as crawler
from multiprocessing import Process, Queue
from twisted.internet import reactor
from google.colab import files
 
 
al=[]
bl=[]
cl=[]
dl=[]
 
#our spider
class QuotesSpider(scrapy.Spider):
    name = "news"
    start_urls = ['https://www.energy.gov/listings/energy-news/']
    
 
 
    def parse(self, response):
      for article_date in response.css('.date::text').extract():
          a=(article_date)
          al.append(a)
      for article_headline in response.css('.title-link::text').extract():
          b=(article_headline)
          bl.append(b)
      for article_description in response.css('.odd::text').extract():
          c=(article_description)
          cl.append(c)
      for article_url in response.css('.title-link::attr(href)').extract():
          d=('https://www.energy.gov'+article_url)
          dl.append(d)
      print(al,bl,cl,dl)
      xl(al,bl,cl,dl)
 
 
 
 
 
# the wrapper to make it run more times
def run_spider(spider):
    def f(q):
        try:
            runner = crawler.CrawlerRunner()
            deferred = runner.crawl(spider)
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run()
            q.put(None)
        except Exception as e:
            q.put(e)
 
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    result = q.get()
    p.join()
 
    if result is not None:
        raise result

def excel():
  print("hello")
  import xlsxwriter
  from google.colab import files

  # Create a workbook and add a worksheet.
  workbook = xlsxwriter.Workbook('Expenses0.xlsx')
  worksheet = workbook.add_worksheet()

  # Some data we want to write to the worksheet.
  expenses = (
      ['Rent', 1000],
      ['Gas',   100],
      ['Food',  300],
      ['Gym',    50],
  )

  # Start from the first cell. Rows and columns are zero indexed.
  row = 0
  col = 0

  # Iterate over the data and write it out row by row.
  for item, cost in (expenses):
      worksheet.write(row, col,     item)
      worksheet.write(row, col + 1, cost)
      row += 1

  # Write a total using a formula.
  worksheet.write(row, 0, 'Total')
  worksheet.write(row, 1, '=SUM(B1:B4)')
  files.download('Expenses0.xlsx')

  workbook.close()
  return 0