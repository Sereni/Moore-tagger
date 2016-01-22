__author__ = ''

import lxml.html
import lxml.etree
import os


def get_links(dirname):
    links = []
    for fname in os.listdir(dirname):
        with open(fname, 'r', encoding='utf-8') as f:
            html = f.read()
        root = lxml.html.fromstring(html)
        posts = root.xpath(u'//*[contains(@class, "post__content")]/h2/a/@href')
        links += ['http://www.hse.ru' + i for i in posts if i.startswith('/news')]
    return links


def get_text(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        html = f.read()
    root = lxml.html.fromstring(html)
    posts = root.xpath(u'//*[contains(@class, "post-title")]/text()')
    posts += root.xpath(u'//*[contains(@class, "post__text")]/p/text()')
    return posts

for i in get_text('news.html'):
    print(i)