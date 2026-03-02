import QtQuick
import QtQuick.Layouts
import "../styles" as Styles

Item {
    id: bubble
    width: parent ? parent.width : 400
    height: row.height + 12

    property string role: "user"
    property string content: ""
    property bool isUser: role === "user"

    RowLayout {
        id: row
        anchors.left: isUser ? undefined : parent.left
        anchors.right: isUser ? parent.right : undefined
        anchors.leftMargin: isUser ? 64 : Styles.Theme.paddingMedium
        anchors.rightMargin: isUser ? Styles.Theme.paddingMedium : 64
        spacing: Styles.Theme.spacingSmall
        layoutDirection: isUser ? Qt.RightToLeft : Qt.LeftToRight

        // Avatar
        Rectangle {
            width: 32; height: 32; radius: Styles.Theme.borderRadius
            color: isUser ? Styles.Theme.primaryColor : Styles.Theme.bgInput
            Layout.alignment: Qt.AlignTop

            Text {
                anchors.centerIn: parent
                text: isUser ? "U" : "AI"
                color: isUser ? Styles.Theme.textOnAccent : Styles.Theme.textPrimary
                font.pixelSize: Styles.Theme.fontSizeSmall
                font.weight: Font.DemiBold
            }
        }

        // Message bubble
        Rectangle {
            Layout.maximumWidth: bubble.width * 0.72
            implicitWidth: Math.min(msgText.implicitWidth + 24, bubble.width * 0.72)
            implicitHeight: msgText.implicitHeight + 18
            radius: Styles.Theme.borderRadiusLarge
            color: isUser ? Styles.Theme.userBubbleColor : Styles.Theme.assistantBubbleColor
            border.color: isUser ? "#305D8A" : Styles.Theme.border
            border.width: 1

            Text {
                id: msgText
                anchors.fill: parent
                anchors.margins: 10
                text: bubble.content
                wrapMode: Text.Wrap
                color: isUser ? Styles.Theme.userBubbleText : Styles.Theme.assistantBubbleText
                font.pixelSize: Styles.Theme.fontSizeMedium
                textFormat: Text.PlainText
                lineHeight: 1.4
            }
        }
    }
}
