import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Popup {
    id: popup
    width: 340
    height: 400
    modal: true
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    dim: true

    Overlay.modal: Rectangle { color: "#80000000" }

    property real temperature: 1.0
    property real topP: 1.0
    property int topK: 0
    property real repeatPenalty: 1.0

    signal settingsApplied(real temp, real tp, int tk, real rp)

    background: Rectangle {
        color: Styles.Theme.bgPopup
        border.color: Styles.Theme.border
        radius: Styles.Theme.borderRadiusLarge
    }

    contentItem: ColumnLayout {
        spacing: 0

        // Header
        Rectangle {
            Layout.fillWidth: true; Layout.preferredHeight: 40; color: "transparent"
            Text {
                anchors.centerIn: parent
                text: qsTr("Sampling Settings")
                font.pixelSize: Styles.Theme.fontSizeLarge; font.weight: Font.DemiBold; color: Styles.Theme.textBright
            }
            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
        }

        // Sliders
        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: Styles.Theme.paddingLarge
            spacing: Styles.Theme.paddingMedium

            // Temperature
            ColumnLayout { Layout.fillWidth: true; spacing: Styles.Theme.spacingTiny
                RowLayout {
                    Text { text: qsTr("Temperature"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                    Item { Layout.fillWidth: true }
                    Text { text: tempSlider.value.toFixed(2); font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.accentColor; font.weight: Font.DemiBold }
                }
                Slider { id: tempSlider; Layout.fillWidth: true; from: 0; to: 2; value: popup.temperature; stepSize: 0.05
                    palette.dark: Styles.Theme.bgInput; palette.mid: Styles.Theme.accentColor
                }
            }

            // Top P
            ColumnLayout { Layout.fillWidth: true; spacing: Styles.Theme.spacingTiny
                RowLayout {
                    Text { text: qsTr("Top P"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                    Item { Layout.fillWidth: true }
                    Text { text: topPSlider.value.toFixed(2); font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.accentColor; font.weight: Font.DemiBold }
                }
                Slider { id: topPSlider; Layout.fillWidth: true; from: 0; to: 1; value: popup.topP; stepSize: 0.05
                    palette.dark: Styles.Theme.bgInput; palette.mid: Styles.Theme.accentColor
                }
            }

            // Top K
            ColumnLayout { Layout.fillWidth: true; spacing: Styles.Theme.spacingTiny
                RowLayout {
                    Text { text: qsTr("Top K"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                    Item { Layout.fillWidth: true }
                    Text { text: topKSlider.value.toFixed(0); font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.accentColor; font.weight: Font.DemiBold }
                }
                Slider { id: topKSlider; Layout.fillWidth: true; from: 0; to: 100; value: popup.topK; stepSize: 1
                    palette.dark: Styles.Theme.bgInput; palette.mid: Styles.Theme.accentColor
                }
            }

            // Repeat Penalty
            ColumnLayout { Layout.fillWidth: true; spacing: Styles.Theme.spacingTiny
                RowLayout {
                    Text { text: qsTr("Repeat Penalty"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                    Item { Layout.fillWidth: true }
                    Text { text: rpSlider.value.toFixed(2); font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.accentColor; font.weight: Font.DemiBold }
                }
                Slider { id: rpSlider; Layout.fillWidth: true; from: 1; to: 2; value: popup.repeatPenalty; stepSize: 0.05
                    palette.dark: Styles.Theme.bgInput; palette.mid: Styles.Theme.accentColor
                }
            }
        }

        Item { Layout.fillHeight: true }

        // Footer
        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: Styles.Theme.border }
        Item {
            Layout.fillWidth: true; Layout.preferredHeight: 44
            Rectangle {
                anchors.centerIn: parent
                width: applyLbl.implicitWidth + 32; height: 30; radius: Styles.Theme.borderRadius
                color: applyMa.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor
                Text { id: applyLbl; anchors.centerIn: parent; text: qsTr("Save"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textOnAccent }
                MouseArea {
                    id: applyMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                    onClicked: { popup.settingsApplied(tempSlider.value, topPSlider.value, Math.round(topKSlider.value), rpSlider.value); popup.close() }
                }
            }
        }
    }
}
