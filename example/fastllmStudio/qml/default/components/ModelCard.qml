import QtQuick
import QtQuick.Layouts
import "../styles" as Styles

Rectangle {
    id: card
    width: parent ? parent.width : 400
    height: 88
    radius: Styles.Theme.borderRadius
    color: cardMa.containsMouse ? Styles.Theme.bgCardHover : Styles.Theme.bgCard
    border.color: running ? Styles.Theme.successColor : Styles.Theme.border
    border.width: 1

    Behavior on color { ColorAnimation { duration: Styles.Theme.animDuration } }

    property string modelId: ""
    property string modelName: ""
    property string modelPath: ""
    property bool running: false
    property int modelPort: 0
    property string device: ""
    property string apiModelName: ""
    property int configPort: 0

    signal settingsClicked(string mid)
    signal startClicked(string mid)
    signal stopClicked(string mid)
    signal removeClicked(string mid)

    MouseArea {
        id: cardMa
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: Styles.Theme.paddingMedium
        anchors.rightMargin: Styles.Theme.paddingMedium
        anchors.topMargin: Styles.Theme.paddingSmall
        anchors.bottomMargin: Styles.Theme.paddingSmall
        spacing: Styles.Theme.spacing

        // Status dot
        Rectangle {
            Layout.preferredWidth: 8
            Layout.preferredHeight: 8
            radius: 4
            color: card.running ? Styles.Theme.successColor : Styles.Theme.textDisabled
            Layout.alignment: Qt.AlignVCenter
        }

        // Info column
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 2

            Text {
                text: card.modelName
                font.pixelSize: Styles.Theme.fontSizeLarge
                font.weight: Font.DemiBold
                color: Styles.Theme.textBright
                elide: Text.ElideRight
                Layout.fillWidth: true
            }

            Text {
                text: card.modelPath
                font.pixelSize: Styles.Theme.fontSizeTiny
                color: Styles.Theme.textSecondary
                elide: Text.ElideMiddle
                Layout.fillWidth: true
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: Styles.Theme.spacingSmall

                Rectangle {
                    Layout.preferredWidth: statusLabel.implicitWidth + 12
                    Layout.preferredHeight: 18
                    radius: 9
                    color: card.running ? Styles.Theme.runningBadgeBg : Styles.Theme.stoppedBadgeBg
                    Text {
                        id: statusLabel
                        anchors.centerIn: parent
                        text: card.running
                              ? qsTr("Running :%1").arg(card.modelPort)
                              : qsTr("Stopped")
                        font.pixelSize: Styles.Theme.fontSizeTiny
                        color: card.running ? Styles.Theme.runningBadgeText : Styles.Theme.stoppedBadgeText
                    }
                }

                Text {
                    text: (card.apiModelName || card.modelName) + "  " + (card.device || "auto")
                    font.pixelSize: Styles.Theme.fontSizeTiny
                    color: Styles.Theme.textDisabled
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
            }
        }

        // Action buttons
        Row {
            spacing: Styles.Theme.spacingSmall
            Layout.alignment: Qt.AlignVCenter

            // Settings
            Rectangle {
                width: 28; height: 28; radius: Styles.Theme.borderRadius
                color: settMa.containsMouse ? Styles.Theme.bgInput : "transparent"
                Text { anchors.centerIn: parent; text: "⚙"; font.pixelSize: 14; color: Styles.Theme.textSecondary }
                MouseArea { id: settMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: card.settingsClicked(card.modelId) }
            }

            // Start / Stop
            Rectangle {
                width: btnLabel.implicitWidth + 20; height: 28; radius: Styles.Theme.borderRadius
                color: card.running
                       ? (stopMa.containsMouse ? "#5A2020" : "#3C1E1E")
                       : (stopMa.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor)
                Text {
                    id: btnLabel; anchors.centerIn: parent
                    text: card.running ? qsTr("Stop") : qsTr("Start")
                    font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.textOnAccent
                }
                MouseArea {
                    id: stopMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        onClicked: card.running ? card.stopClicked(card.modelId) : card.startClicked(card.modelId)
                }
            }

            // Remove
            Rectangle {
                width: 28; height: 28; radius: Styles.Theme.borderRadius
                color: rmMa.containsMouse ? "#3C1E1E" : "transparent"
                Text { anchors.centerIn: parent; text: "✕"; font.pixelSize: 12; color: Styles.Theme.textSecondary }
                MouseArea { id: rmMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: card.removeClicked(card.modelId) }
            }
        }
    }
}
