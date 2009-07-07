#include "osgCuda/Texture"

namespace osgCuda
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    template<>
    bool Texture<unsigned char>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_UNSIGNED_BYTE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec4ub>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGBA );
        asTexture()->setSourceFormat( GL_RGBA );
        asTexture()->setSourceType( GL_UNSIGNED_BYTE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<char>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_BYTE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec2b>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceType( GL_BYTE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec3b>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGB );
        asTexture()->setSourceFormat( GL_RGB );
        asTexture()->setSourceType( GL_BYTE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec4b>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGBA );
        asTexture()->setSourceFormat( GL_RGBA );
        asTexture()->setSourceType( GL_BYTE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<unsigned short>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_UNSIGNED_SHORT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<short>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_SHORT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec2s>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceType( GL_SHORT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec3s>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGB );
        asTexture()->setSourceFormat( GL_RGB );
        asTexture()->setSourceType( GL_SHORT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec4s>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGBA );
        asTexture()->setSourceFormat( GL_RGBA );
        asTexture()->setSourceType( GL_SHORT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<unsigned int>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_UNSIGNED_INT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<int>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_INT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<float>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_FLOAT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec2f>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceType( GL_FLOAT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec3f>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGB );
        asTexture()->setSourceFormat( GL_RGB );
        asTexture()->setSourceType( GL_FLOAT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec4f>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGBA );
        asTexture()->setSourceFormat( GL_RGBA );
        asTexture()->setSourceType( GL_FLOAT );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<double>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE );
        asTexture()->setSourceFormat( GL_LUMINANCE );
        asTexture()->setSourceType( GL_DOUBLE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec2d>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceFormat( GL_LUMINANCE_ALPHA );
        asTexture()->setSourceType( GL_DOUBLE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec3d>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGB );
        asTexture()->setSourceFormat( GL_RGB );
        asTexture()->setSourceType( GL_DOUBLE );

        return true;
    }

    //------------------------------------------------------------------------------
    template<>
    bool Texture<osg::Vec4d>::initFormatAndType()
    {
        asTexture()->setInternalFormat( GL_RGBA );
        asTexture()->setSourceFormat( GL_RGBA );
        asTexture()->setSourceType( GL_DOUBLE );

        return true;
    }

}

