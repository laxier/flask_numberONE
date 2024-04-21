This Flask program is written in Python and contains routes for handling user authentication, posting, commenting, and deleting posts. 

1. Routes:
- Home Page (/, /index): Requires user authentication. Users can create posts with an optional image upload feature. 
- Login (/login)
- Registration (/register): register with a username, email, and password.
- Logout (/logout)
- Delete Post (/delete/<int:id>): Allows authorized users to delete their posts. Admins can delete any post.
- Delete Comment (/post/<int:commID>/delete/<int:id>): Allows authorized users to delete comments on posts.
- Comment Post (/post/<int:id>): Allows users to comment on specific posts.

HTTP pages include styling elements, header handling, link styles, images and navigation menus.
The Bootstrap and Moment.js libraries are used to improve the appearance and functionality of the page.
- The program uses SQLAlchemy for database operations.
- User authentication is implemented using Flask-Login.

![bandicam 2024-04-21 15-22-33-340](https://github.com/laxier/flask_numberONE/assets/66477335/cae142d1-345b-46ef-9f69-e2d651f8957f)
![bandicam 2024-04-21 15-21-52-334](https://github.com/laxier/flask_numberONE/assets/66477335/7324b5d6-c645-4fa9-b9a1-e684af4f0371)
![bandicam 2024-04-21 15-21-31-550](https://github.com/laxier/flask_numberONE/assets/66477335/f0c6fcef-0883-4c99-9790-8b5ac5d92b75)
