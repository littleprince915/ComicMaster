from . import db


class Image (db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    filename = db.Column(db.String(50), unique=True, nullable=True)
    data = db.Column(db.LargeBinary)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.acctNo'))

def createimage(filename, data, user):
    newFile = Image(filename=file.filename, data=file.read(), owner=user)
    db.session.add(newFile)
    db.session.commit()

    #TODO: call the text detection function

    return newFile.id


def getuserimages(userid):
    #TODO: return the imageids of the user

    return [1,1,1]


def getimage(imageid):
    #TODO: return the image boxes and translation

    return None


def getimagefile(imageid):
    #TODO: return the imagefile

    return None