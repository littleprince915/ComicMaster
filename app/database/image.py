from sqlalchemy.orm import load_only
from app.database.user import getuserbyid
from app import db

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    data = db.Column(db.LargeBinary)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.acctNo'))


def createimage(data, userid):
    #TODO: call the text detection function

    #TODO: make 3 sizes. original_size, gallery_size, translation_size
    user = getuserbyid(userid)
    print user
    newFile = Image(data=data, owner=user)
    db.session.add(newFile)
    db.session.commit()

    return newFile.id


def getuserimageids(userid):
    #TODO: return the imageids of the user
    # imageids = db.session.query(Image).filter(Image.owner_id == userid).all()
    imageids = Image.query.filter_by(owner_id=userid).all()
    imageids = [x.id for x in imageids]
    print imageids
    return imageids


def getimage(imageid):
    #TODO: return the image boxes and translation

    return []


def getimagefile(imageid):
    #TODO: return the imagefile
    imagedata = Image.query.get(int(imageid))

    return imagedata.data