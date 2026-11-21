import pytest
from smart_overlay.pdf_pages import parse_pdf_pages, format_pdf_pages


@pytest.mark.parametrize('spec,expected', [('', list(range(1, 11))), (' \t ', list(range(1, 11))), (' AlL ', list(range(1, 11))), (' 6, 1 - 3,2,9-10 ', [1,2,3,6,9,10])])
def test_parse(spec, expected):
    assert parse_pdf_pages(spec, 10) == expected


@pytest.mark.parametrize('spec', [',1','1,','1,,2','0','-1','3-2','1-11','11','1-','1:3','1.0','all,1','1--2','١'])
def test_invalid(spec):
    with pytest.raises(ValueError):
        parse_pdf_pages(spec, 10)


def test_format():
    assert format_pdf_pages([9, 1, 3, 2, 6, 8, 7, 6]) == '01-03,06-09'
    assert format_pdf_pages([100, 1]) == '01,100'
    assert format_pdf_pages([]) == ''


@pytest.mark.parametrize('total', [0, -1])
def test_invalid_total(total):
    with pytest.raises(ValueError):
        parse_pdf_pages('', total)
