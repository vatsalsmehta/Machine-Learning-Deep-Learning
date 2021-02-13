# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scrapy

class NewsSpider(scrapy.Spider):
    name='news'
    
    start_urls=['https://www.energy.gov/listings/energy-news']
    
    def parse(self,response):
        for news in response.xpath("//div[@class='field-items']"):
            yield{
                    'news_text':news.xpath(".//div[@class='field-item odd']" ).extract_first()
                    }
        
        next_page=response.xpath("//li[@class=1 last active]/a/@href").extract_first()
        if next_page is not None:
            next_page_link=response.urljoin(next_page)
            yield scrapy.Request(url=next_page_link,callback=self.parse)