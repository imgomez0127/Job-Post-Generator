from requests_html import HTMLSession, HTML
doc = """<a href='https://httpbin.org'>"""
html = HTML(html=doc)
script = """
    () => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }
"""
val = html.render(script=script, reload=False)

print(html.html)
print(val)
