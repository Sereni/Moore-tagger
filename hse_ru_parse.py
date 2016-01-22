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
        with open(os.path.join(dirname,fname), 'r', encoding='utf-8') as f:
            html = f.read()
        root = lxml.html.fromstring(html)
        posts = root.xpath(u'//*[contains(@class, "post__content")]/h2/a/@href')
        links += ['http://www.hse.ru' + i for i in posts if i.startswith('/news')]
    return links


def get_text(fname): # неа, эта штука только на половине статей работает =(( ну ее н
    '''
    :param fname: путь к хтмл файлу с полной версией новостной статьи
    :return: текст статьи (строка)
    '''
    with open(fname, 'r', encoding='utf-8') as f:
        html = f.read()
    root = lxml.html.fromstring(html)
    posts = root.xpath(u'//*[contains(@class, "post-title")]/text()') + ['\n']
    posts += root.xpath(u'//*[contains(@class, "post__text")]/p/text()')
    if not posts:
        posts = root.xpath(u'//*[contains(@class, "title")]/p/text()') + ['\n']
        posts += root.xpath(u'//*[contains(@class, "text q lead-in")]/text()')
    return ' '.join(posts)


# DIR = 'pages'  # тут путь к папке со страницами типа http://www.hse.ru/news/page5.html
# with open('all_links.txt', 'w', encoding='utf-8') as f:  # напишем ссылки в all_links.txt
#     f.write('\n'.join(get_links(DIR)))

# DIR2 = 'articles'  # тут путь к папке с полными статьями
# for fname in os.listdir(DIR2):
#     print(get_text(os.path.join(DIR2,fname)))