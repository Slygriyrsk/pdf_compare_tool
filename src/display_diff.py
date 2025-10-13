import difflib

#compare the texts and return diff in html format or we can also use JSON may be in future
def cmptxt(old_text: str, new_text: str) -> str:
    differ = difflib.HtmlDiff(wrapcolumn=100)
    return differ.make_file(
        old_text.splitlines(),
        new_text.splitlines(),
        fromdesc="Old Version",
        todesc="New Version"
    )