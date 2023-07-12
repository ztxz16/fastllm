package com.doujiao.xiaozhihuiassistant.utils;

import android.content.ContentUris;
import android.content.Context;
import android.content.CursorLoader;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.support.annotation.RequiresApi;

import java.io.File;

public class UriUtils {
    /**
     * 从Uri获取文件路径，存储路径需要获取
     * android.permission.READ_EXTERNAL_STORAGE权限
     * 适用于MediaStore和其他基于文件的内容提供
     *
     * @param context 上下文对象
     * @param uri     Uri
     */
    public static String getPath(Context context, Uri uri) {
        int SDK_INT = Build.VERSION.SDK_INT;
        if(SDK_INT < Build.VERSION_CODES.HONEYCOMB) {
            // No longer supported
            throw new RuntimeException("SDK_INT=" + SDK_INT);
        } else if(SDK_INT < Build.VERSION_CODES.KITKAT) {
            return getPathFromUri_API11to18(context, uri);
        } else // if(SDK_INT >= Build.VERSION_CODES.KITKAT)
            return getPathFromUri_API19(context, uri);
    }

    private static String getPathFromUri_API11to18(Context context, Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        String result = null;

        CursorLoader cursorLoader = new CursorLoader(context, uri, projection, null, null, null);
        Cursor cursor = cursorLoader.loadInBackground();

        if(cursor != null) {
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            result = cursor.getString(column_index);
            cursor.close();
        }
        return result;
    }

    @RequiresApi(Build.VERSION_CODES.KITKAT)
    private static String getPathFromUri_API19(Context context, Uri uri) {
        String scheme = uri.getScheme();
        String authority = uri.getAuthority();
        if(scheme == null)
            return null;

        // https://stackoverflow.com/questions/18263489/why-doesnt-string-switch-statement-support-a-null-case
        // https://stackoverflow.com/questions/27251456/start-browser-via-intent-url-with-schema-http-uppercase-error
        scheme = scheme.toLowerCase();  // For resiliency, since RFC 2396 says scheme names are lowercase.

        if(DocumentsContract.isDocumentUri(context, uri)) {
            if("com.android.externalstorage.documents".equals(authority)) {  // ExternalStorageProvider
                final String documentId = DocumentsContract.getDocumentId(uri);
                final String[] split = documentId.split(":");
                final String type = split[0];

                if ("primary".equalsIgnoreCase(type)) {
                    return new File(Environment.getExternalStorageDirectory(), split[1]).getPath();
                }

                // TODO handle non-primary volumes
            } else if("com.android.providers.downloads.documents".equals(authority)) {  // DownloadsProvider
                return getDataColumn(context, uri, null/* selection*/, null/* selectionArgs */);
            } else if("com.android.providers.media.documents".equals(authority)) {  // MediaProvider
                final String documentId = DocumentsContract.getDocumentId(uri);
                final String[] split = documentId.split(":");
                final String type = split[0];

                final Uri contentUri;
                if("image".equals(type)) {
                    contentUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
                } else if("video".equals(type)) {
                    contentUri = MediaStore.Video.Media.EXTERNAL_CONTENT_URI;
                } else if("audio".equals(type)) {
                    contentUri = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI;
                } else {
                    contentUri = null;
                }

                final String selection = "_id=?";
                final String[] selectionArgs = new String[] { split[1] };
                return getDataColumn(context, contentUri, selection, selectionArgs);
            }
        } else if("content".equals(scheme)) {  //  MediaStore for most cases
            // content://com.google.android.apps.photos.contentprovider/0/1/content%3A%2F%2Fmedia%2Fexternal%2Fimages%2Fmedia%2F75209/ACTUAL
            if ("com.google.android.apps.photos.contentprovider".equals(uri.getAuthority()))
                return uri.getLastPathSegment();
            else
                return getDataColumn(context, uri, null/* selection*/, null/* selectionArgs */);
        } else if("file".equals(scheme)) {  // file
            return uri.getPath();
        }

        return null;
    }

    /**
     * Get the value of the data column for this Uri. This is useful for MediaStore Uris, and other
     * file-based ContentProviders.
     *
     * @param context       The context.
     * @param uri           The Uri to query.
     * @param selection     (Optional) Filter used in the query.
     * @param selectionArgs (Optional) Selection arguments used in the query.
     * @return The value of the _data column, which is typically a file path.
     */
    private static String getDataColumn(Context context, Uri uri, String selection, String[] selectionArgs) {
        // Though MediaStore.Video.Media.DATA is used here, actually "_data" is the key. So DATA
        // here can be MediaStore.Images.Media.DATA or MediaStore.Audio.Media.DATA, etc.
        final String DATA = MediaStore.Video.Media.DATA;
        final String[] projection = { DATA };
        Cursor cursor = context.getContentResolver().query(uri, projection, selection, selectionArgs, null/* sortOrder */);
        if(cursor != null) {
            cursor.moveToFirst();
            int index = cursor.getColumnIndex(DATA);
//			if(index >= 0)
            String data = cursor.getString(index);
            cursor.close();
            return data;
        }
        return null;
    }

}
