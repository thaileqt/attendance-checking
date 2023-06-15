from st_pages import Page, show_pages

# Optional -- adds the title and icon to the current page
# add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("pages/recognize.py", "Check attendance"),
        Page("pages/add_faces.py", "Add face"),
    ]
)

