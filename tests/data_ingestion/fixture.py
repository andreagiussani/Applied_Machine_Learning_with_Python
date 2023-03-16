from io import StringIO


def get_mocked_string_csv():
    return StringIO(
        """col1,col2,col3,y
        1,4.4,99,1
        2,4.5,200,1
        3,4.7,65,0
        4,1.5,140,0"""
    )
