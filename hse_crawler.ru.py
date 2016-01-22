__author__ = ''

import lxml.html
import lxml.etree
import os


def get_links(dirname):
    '''
    :param dirname: папка с хтмл файлами, где хранятся страницы типа http://www.hse.ru/news/page5.html (списки статей с краткими аннотациями и ссылками на полные версии)
    :return: links - массив всех ссылок на полные версии статей (только те, которые на портале hse.ru, перенаправления на другие порталы исключаются)
    '''
    links = []
    for fname in os.listdir(dirname):
        with open(fname, 'r', encoding='utf-8') as f:
            html = f.read()
        root = lxml.html.fromstring(html)
        posts = root.xpath(u'//*[contains(@class, "post__content")]/h2/a/@href')
        links += ['http://www.hse.ru' + i for i in posts if i.startswith('/news')]
    return links


def get_text(fname):
    '''
    :param fname: путь к хтмл файлу с полной версией новостной статьи
    :return: текст статьи (строка)
    '''
    with open(fname, 'r', encoding='utf-8') as f:
        html = f.read()
    root = lxml.html.fromstring(html)
    posts = root.xpath(u'//*[contains(@class, "post-title")]/text()') + ['\n']
    posts += root.xpath(u'//*[contains(@class, "post__text")]/p/text()')
    return ' '.join(posts)

print (get_text('news.html'))