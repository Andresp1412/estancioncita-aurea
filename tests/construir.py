"""Ensambla index.html (producción) y _local.html (librerías locales para pruebas)."""
import os, re
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
head = open(os.path.join(R, 'src', 'head.html')).read()
html = head + open(os.path.join(R, 'src', 'extra.css')).read() + '</style>\n</head>\n<body>\n<div id="root"></div>\n\n<script>\n' + open(os.path.join(R, 'src', 'app.js')).read() + '</script>\n</body>\n</html>\n'
open(os.path.join(R, 'index.html'), 'w').write(html)
loc = (html.replace('https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js', '/nm/chart.js/dist/chart.umd.js')
           .replace('https://cdnjs.cloudflare.com/ajax/libs/react/18.3.1/umd/react.production.min.js', '/nm/react/umd/react.production.min.js')
           .replace('https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.3.1/umd/react-dom.production.min.js', '/nm/react-dom/umd/react-dom.production.min.js'))
loc = re.sub(r'<link rel="preconnect"[^>]*>\s*<link rel="stylesheet"[^>]*>', '', loc)
open(os.path.join(R, '_local.html'), 'w').write(loc)   # copia de pruebas con librerías locales (no se publica)
print('index.html', len(html), 'bytes')
